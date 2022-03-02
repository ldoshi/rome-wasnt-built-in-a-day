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
import shutil
import pickle
import os
import pathlib
import time
from collections.abc import Hashable

from bridger.logging import log_entry


class ObjectLogManager:
    """Provides a unified interface to log pickle-able objects."""

    def __init__(self, dirname: str):
        """Creates directory dirname to store logs.

        Clears the contents of the directory if the dirname existed previously.

        Args:
          dirname: The name of the directory to create.
        """
        self._dirname = dirname
        shutil.rmtree(self._dirname, ignore_errors=True)
        path = pathlib.Path(self._dirname)
        path.mkdir(parents=True, exist_ok=True)

        self._object_loggers = {}
        self._object_logger_costs = collections.defaultdict(float)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for log_filename, object_logger in self._object_loggers.items():
            start = time.perf_counter()
            object_logger.close()
            self._object_logger_costs[log_filename] += (time.perf_counter() - start)

        print("EXITING")
        for log_filename, cost in self._object_logger_costs.items():
            print(f"{log_filename}: {cost}")

    def log(self, log_filename: str, log_entry: Any) -> None:
        """Logs the provided entry to a log file named log_filename.

        Args:
          log_filename: A unique label describing the log in which to place
            log_entry. This label is also the actual log filename.
          log_entry: The object to be logged.
        """
        if log_filename not in self._object_loggers:
            self._object_loggers[log_filename] = ObjectLogger(
                dirname=self._dirname, log_filename=log_filename
            )

        start = time.perf_counter()
        self._object_loggers[log_filename].log(log_entry)
        self._object_logger_costs[log_filename] += (time.perf_counter() - start)


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
        if make_hashable_fn:
            self._make_hashable_fn = make_hashable_fn
        else:
            self._make_hashable_fn = lambda x: x
        self._normalizer = {}

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
        self._object_log_manager.log(
            self._log_filename,
            log_entry.NormalizedLogEntry(id=object_id, object=object),
        )
        self._normalizer[hashable_object] = object_id
        return object_id


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
        if make_hashable_fn:
            self._make_hashable_fn = make_hashable_fn
        else:
            self._make_hashable_fn = lambda x: x
        self._normalizer = {}

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
        self._object_log_manager.log(
            self._log_filename,
            log_entry.NormalizedLogEntry(id=object_id, object=object),
        )
        self._normalizer[hashable_object] = object_id
        return object_id


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
        self._log_file = open(os.path.join(dirname, log_filename), "wb")

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


def read_object_log(dirname: str, log_filename: str):
    with open(os.path.join(dirname, log_filename), "rb") as f:
        buffer = None
        while True:
            try:
                buffer = pickle.load(f)

                for element in buffer:
                    yield element

            except EOFError:
                break
