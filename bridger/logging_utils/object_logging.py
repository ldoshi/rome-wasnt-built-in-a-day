"""Logs pickle-able objects for analysis and debugging.

To log a new entity, define a LogEntry object in log_entry.py.

Standard users will not instantiate ObjectLogger directly.

Usage:
  with ObjectLogManager(dirname) as logger:
    logger.log("history", training_history_event)
    logger.log("buffer", buffer_event)

"""
from typing import Any, Callable, Optional

import collections
import copy
import shutil
import pickle
import os
import pathlib
import torch

from collections.abc import Hashable

from bridger.logging_utils import log_entry
from bridger.logging_utils.object_log_readers import read_object_log


class ObjectLogManager:
    """Provides a unified interface to log pickle-able objects."""

    def __init__(self, object_logging_base_dir: str, experiment_name: str):
        """Creates directory dirname to store logs.

        Clears the contents of the directory if the dirname existed previously.

        Args:
          object_logging_base_dir: The name of the directory to create.
          experiment_name: The name of the experiment to create.
        """
        self._base_dir = object_logging_base_dir
        self._experiment_name = experiment_name
        self._experiment_dir = os.path.join(self._base_dir, self._experiment_name)
        shutil.rmtree(self._experiment_dir, ignore_errors=True)
        path = pathlib.Path(self._experiment_dir)
        path.mkdir(parents=True, exist_ok=True)

        self._object_loggers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for object_logger in self._object_loggers.values():
            object_logger.close()

    def log(
        self, log_filename: str, log_entry: Any, log_in_parent_dir: bool = False
    ) -> None:
        """Logs the provided entry to a log file named log_filename.

        Args:
          log_filename: A unique label describing the log in which to place
            log_entry. This label is also the actual log filename.
          log_entry: The object to be logged.
          log_in_parent_dir: A boolean describing whether to log in the base dir
          instead of the experiment dir.
        """
        if log_filename not in self._object_loggers:
            if log_in_parent_dir:
                self._object_loggers[log_filename] = ObjectLogger(
                    dirname=self._base_dir,
                    log_filename=log_filename,
                    buffer_size = 1,
                )
            else:
                self._object_loggers[log_filename] = ObjectLogger(
                    dirname=self._experiment_dir,
                    log_filename=log_filename,
                )

        self._object_loggers[log_filename].log(log_entry)

    def read_base_dir_log_file(self, log_filename: str) -> list[Any]:
        """Reads a log entry file in the base directory and returns a list of log entries.

        Args:
            log_filename: The log filename describing the log to read in the base directory.
        """
        return list(read_object_log(os.path.join(self._base_dir, log_filename)))


# TODO(arvind): Refactor out common code shared between
# LoggerAndNormalizer and OccurrenceLogger.
class LoggerAndNormalizer:
    """Logs objects and normalizes them to unique ids.

    Some objects may be logged very frequently and may be expensive to
    log. One example is the state representation. This tool logs such
    objects exactly once and returns a unique id for each object to be
    used in the other frequent log entries.
    """

    def __init__(
        self,
        log_filename: str,
        object_log_manager: ObjectLogManager,
        log_entry_object_class: Any,
        make_hashable_fn: Optional[Callable[[Any], Hashable]] = None,
        read_existing_log_entry: bool = False,
    ):
        """Store logging directives.

        Args:
          log_filename: A unique label describing the log in which to
            place log_entry. This label is also the actual log
            filename.
          object_log_manager: Logger for pickle-able objects.
          log_entry_object_class: The required type of the object
            passed to get_logged_object_id in this instance of
            LoggerAndNormalizer. The type check is conducted at runtime
            with each get_logged_object_id call.
          make_hashable_fn: A function that converts the objects to be
            logged into something that can be hashed. The default
            value is the identity function.

        """
        self._log_filename = log_filename
        self._object_log_manager = object_log_manager
        self._log_entry_object_class = log_entry_object_class

        if self._log_entry_object_class is torch.Tensor and not make_hashable_fn:
            # Explicitly error on a known source of unexpected
            # behavior.
            raise ValueError(
                f"Hashing torch.Tensor directly does not produce desired"
                f"results. Tensors which are torch.equal may still have "
                f"different hash values. Please provide an argument for "
                f"make_hashable_fn."
            )

        if make_hashable_fn:
            self._make_hashable_fn = make_hashable_fn
        else:
            self._make_hashable_fn = lambda x: x

        # Object ids are assigned sequentially so they can be used
        # directly as indices for the reverse lookup.
        self._normalizer = {}
        self._normalizer_reverse_lookup = []
        if read_existing_log_entry:
            for log_entry in object_log_manager.read_base_dir_log_file(
                log_filename=log_filename
            ):
                # Object IDs should be presented in sequential order.
                assert len(self._normalizer_reverse_lookup) == log_entry.id

                hashable_object = self._make_hashable_fn(log_entry.object)
                self._normalizer[hashable_object] = log_entry.id
                self._normalizer_reverse_lookup.append(log_entry.object)

    def get_logged_object_id(self, object: Any) -> int:
        """Returns the unique id for the provided object.

        The object must be an instance of the log_entry_object_class
        provided in the init.

        The object will additionally be logged if it has not yet been
        logged.

        The id is only unique within the scope of this execution. The
        id is not a function of the object itself.

        Args:
          object: The object for which to obtain a unique id and log if
            it has not been logged before.

        Returns:
          The unique id associated with the object.
        """
        if not isinstance(object, self._log_entry_object_class):
            raise ValueError(
                f"Provided object of type {type(object)} instead of "
                f"expected type {self._log_entry_object_class}"
            )

        hashable_object = self._make_hashable_fn(object)
        if hashable_object in self._normalizer:
            return self._normalizer[hashable_object]

        object_id = len(self._normalizer)
        object_copy = copy.deepcopy(object)
        # TODO(lyric): Consider adding an init arg as to whether the
        # object should be copied or not. Per PR#104, the copy will be
        # required for data coming from training batches.
        self._object_log_manager.log(
            self._log_filename,
            log_entry.NormalizedLogEntry(id=object_id, object=object_copy),
            log_in_parent_dir=True,
        )
        self._normalizer[hashable_object] = object_id
        assert len(self._normalizer_reverse_lookup) == object_id
        self._normalizer_reverse_lookup.append(object_copy)
        return object_id

    def get_logged_object_by_id(self, object_id: int) -> Any:
        """Returns the logged object corresponding to the provided object_id.

        Args:
          object_id: The id to look up.

        Returns:
          The object corresponding to object_id.

        Raises:
          ValueError: If the object_id cannot be found.
        """
        if object_id < len(self._normalizer_reverse_lookup):
            return self._normalizer_reverse_lookup[object_id]

        raise ValueError(
            f"Requested object id {object_id}  was not produced by "
            "this LoggerAndNormalizer instance"
        )


# TODO(Issue#112): Consider implementing the occurrence metadata
# tracking as a sliding window over the most recent b batches instead
# over all time.
class OccurrenceLogger:

    """Logs the occurrence of an object with its batch_idx.

    The OccurrenceLogger also maintains metadata of how frequently
    each object was logged and supports requests for the top-N logged
    objects by frequency.

    A LoggerAndNormalizer can be optionally provided for efficient
    logging of the provided object.

    """

    def __init__(
        self,
        log_filename: str,
        object_log_manager: ObjectLogManager,
        log_entry_object_class: Any,
        logger_and_normalizer: Optional[LoggerAndNormalizer] = None,
    ):
        """Store logging directives.

        Args:
          log_filename: A unique label describing the log in which to
            place log_entry. This label is also the actual log
            filename.
          object_log_manager: Logger for pickle-able objects.
          log_entry_object_class: The required type of the object
            passed to log_occurrence in this instance of OccurrenceLogger.
            The type check is conducted at runtime with each
            log_occurrence call.
          logger_and_normalizer: Helper to enable efficient logging of the
            object if it's expensive to log.
        """
        self._log_filename = log_filename
        self._object_log_manager = object_log_manager
        self._log_entry_object_class = log_entry_object_class
        self._logger_and_normalizer = logger_and_normalizer

        if (
            self._log_entry_object_class is torch.Tensor
            and not self._logger_and_normalizer
        ):
            # Explicitly error on a known source of unexpected
            # behavior.
            raise ValueError(
                f"Hashing torch.Tensor directly does not produce desired"
                f"results. Tensors which are torch.equal may still have "
                f"different hash values. Options: (1) Determine if a "
                f"logger_and_normalizer is appropriate for the tensor "
                f"and set make_hashable_fn there. (2) Add a init parameter"
                f" for make_hashable_fn to OccurrenceLogger if (1) does not"
                f" apply"
            )

        self._occurrence_tracker = collections.Counter()

    def log_occurrence(self, batch_idx: int, object: Any) -> None:
        """Logs an occurrence for the provided object.

        The object must be an instance of the log_entry_object_class
        provided in the init.

        One of the following is presumed to be true:
          1. The object is safely hashable.
          2. A logger_and_normalizer was provided in the init, which
             converts the object into an int id.

        Args:
          batch_idx: The batch_idx for which to log this occurrence of
            object.
          object: The object whose occurrence is being logged.

        """
        if not isinstance(object, self._log_entry_object_class):
            raise ValueError(
                f"Provided object of type {type(object)} instead of "
                f"expected type {self._log_entry_object_class}"
            )

        if self._logger_and_normalizer:
            object_representation = self._logger_and_normalizer.get_logged_object_id(
                object
            )
        else:
            # TODO(lyric): Consider adding an init arg as to whether
            # the object should be copied or not. Per PR#104, the copy
            # will be required for data coming from training batches.
            object_representation = copy.deepcopy(object)

        self._occurrence_tracker[object_representation] += 1

        self._object_log_manager.log(
            self._log_filename,
            log_entry.OccurrenceLogEntry(
                batch_idx=batch_idx, object=object_representation
            ),
        )

    def get_top_n(self, n=None) -> list[Any]:
        if self._logger_and_normalizer:
            return [
                self._logger_and_normalizer.get_logged_object_by_id(entry[0])
                for entry in self._occurrence_tracker.most_common(n)
            ]
        else:
            return [entry[0] for entry in self._occurrence_tracker.most_common(n)]


# TODO(lyric): Consider changing the buffer size metric to be based on
# size vs entry count.
#
# TODO(lyric): Consider adding enforcement that a given ObjectLogger
# only logs a single type of entry. Currently this is enforced by
# convention.
class ObjectLogger:
    """Logs pickle-able objects for analysis and debugging."""

    def __init__(self, dirname: str, log_filename: str, buffer_size=1000):
        self._buffer_size = buffer_size
        self._buffer = []
        self._log_file = open(os.path.join(dirname, log_filename), "ab")

    def _flush_buffer(self):
        if self._buffer:
            pickle.dump(self._buffer, self._log_file)
        self._buffer = []

    def log(self, log_entry: Any) -> None:
        self._buffer.append(log_entry)

        if len(self._buffer) == self._buffer_size:
            self._flush_buffer()

    def close(self):
        self._flush_buffer()
        self._log_file.close()
