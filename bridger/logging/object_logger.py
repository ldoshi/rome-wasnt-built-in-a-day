"""Logs pickle-able objects for analysis and debugging.

Usage: To log a new entity, define a LogEntry object in log_entry.py.
"""
from typing import Any
import shutil
import pickle
import os

def create_logging_dir(dirname: str) -> None:
    """Creates directory dirname if it doesn't exist.

    Clears the contents of the directory if the dirname existed previously.
    
    Args:
      dirname: The name of the directory to create.
    """
    shutil.rmtree(dirname)
    os.mkdir(dirname)

# TODO(lyric): Consider changing the buffer size metric to be based on
# size vs entry count.
#
# TODO(lyric): Consider adding enforcement that a given ObjectLogger
# only logs a single type of entry. Currently this is enforced by
# convention.
class ObjectLogger():
    """Logs pickle-able objects for analysis and debugging."""
    def __init__(self, dirname: str, log_filename: str, buffer_size=1000):
        self._dirname = dirname
        self._log_filename = log_filename
        self._buffer_size = buffer_size
        self._buffer = []

    def __enter__(self): 
        self._log_file = open(os.path.join(dirname, log_filename), "wb")
        
    def _flush(self):
        pickle.dump(self._buffer, self._log_file)
        self._buffer = []
        
    def log(log_entry: Any):
        self._buffer.append(log_entry)

        if len(self._buffer) == buffer_size:
            self._flush()
        
    def __exit__(self):
        self._flush()
        self._log_file.close()

# make the reader.
