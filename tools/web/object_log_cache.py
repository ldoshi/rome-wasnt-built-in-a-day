"""Cache to load and store objects read in from logs.

The implementation relies on the fact that Flask in development mode
is single-process. Combined with the GIL, that means we do not expect
any race conditions to get/load data.

"""

import collections
import os
import time
import torch

from typing import Any, Callable, Dict, List

from bridger.logging_utils import log_entry
from bridger.logging_utils import object_log_readers

ACTION_INVERSION_DATABASE_KEY = "action_inversion_database_key"
TRAINING_HISTORY_DATABASE_KEY = "training_history_database_key"


def _load_action_inversion_database(
    log_dir: str,
) -> object_log_readers.ActionInversionDatabase:
    return object_log_readers.ActionInversionDatabase(dirname=log_dir)


def _load_training_history_database(
    log_dir: str,
) -> object_log_readers.TrainingHistoryDatabase:
    return object_log_readers.TrainingHistoryDatabase(dirname=log_dir)


_LOADERS = {
    ACTION_INVERSION_DATABASE_KEY: _load_action_inversion_database,
    TRAINING_HISTORY_DATABASE_KEY: _load_training_history_database,
}


class ObjectLogCache:
    """Loads and caches logged objects for efficient re-access.

    Attributes:
      key_hit_counts: Dict of the number of cache hits per key.
      key_miss_counts: Dict of the number of cache misses per key.
    """

    def __init__(
        self, log_dir: str, loaders: Dict[str, Callable[[str], Any]] = _LOADERS
    ):
        """Initializes cache.

        Args:
          log_dir: The directory containing the object log files.
          loaders: A dict containing a custom load function for each
            supported key. On cache miss, the load function will be
            called and its return value will be cached as the value
            corresponding to the key.

        """

        self._log_dir = log_dir
        self._cache = {}
        self._loaders = loaders
        self.key_hit_counts = collections.defaultdict(int)
        self.key_miss_counts = collections.defaultdict(int)

    def get(self, key: str) -> Any:
        """Retrieves the requested data constructed from log entries.

        If the data has not been loaded yet, it will be loaded here
        first. Each support key type has a custom log entry loader.

        Args:
          key: The corresponding to the desired log-based data.

        Returns:
          Logged data loaded into a data structure corresponding to
          the key. See loader definitions. None if backing files do not
          exist at the provided location.

        Raises:
          ValueError on unsupported key.
          FileNotFoundError if key does not correspond to a backing
            data file.

        """
        if key not in self._loaders:
            raise ValueError(f"Unsupported key: {key}")

        if key not in self._cache:
            self.key_miss_counts[key] += 1
            start = int(time.time() * 1e3)
            self._cache[key] = self._loaders[key](self._log_dir)
            end = int(time.time() * 1e3)
            print(f"ObjectLogCache loading {key} took {end-start} ms.")
        else:
            self.key_hit_counts[key] += 1

        return self._cache[key]
