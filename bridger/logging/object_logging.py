"""Logs pickle-able objects for analysis and debugging.

To log a new entity, define a LogEntry object in log_entry.py.

Standard users will not instantiate ObjectLogger directly.

Usage:
  with ObjectLogManager(dirname) as logger:
    logger.log("history", training_history_event)
    logger.log("buffer", buffer_event)

"""
from typing import Any
import shutil
import pickle
import os
import pathlib


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for object_logger in self._object_loggers.values():
            object_logger.close()

    def log(self, log_filename: str, log_entry: Any) -> None:
        """Logs the provided entry to a log file named log_filename.

        Args:
          log_filename: A unique label describing the log in which to place
            log_entry. The label is also the actual log filename.
          log_entry: The object to be logged.
        """
        if log_filename not in self._object_loggers:
            self._object_loggers[log_filename] = ObjectLogger(
                dirname=self._dirname, log_filename=log_filename
            )

        self._object_loggers[log_filename].log(log_entry)


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
