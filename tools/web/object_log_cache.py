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

from bridger.logging import log_entry
from bridger.logging import object_log_readers

STATES_BY_STATE_ID_KEY = "states_by_state_id"
ACTION_INVERSION_REPORTS_BY_BATCH_ID_KEY = "action_inversion_reports_by_batch_id"
TRAINING_HISTORY_DATABASE_KEY = "training_history_database_key"


def _load_action_inversion_reports_by_batch_id(
    log_dir: str,
) -> Dict[int, List[log_entry.ActionInversionReportEntry]]:
    reports = collections.defaultdict(list)

    log_path = os.path.join(log_dir, log_entry.ACTION_INVERSION_REPORT_ENTRY)
    for entry in object_log_readers.read_object_log(log_path):
        reports[entry.batch_idx].append(entry)

    return reports


def _load_states_by_state_id(log_dir: str) -> Dict[int, torch.Tensor]:
    states = {}

    log_path = os.path.join(log_dir, log_entry.STATE_NORMALIZED_LOG_ENTRY)
    for entry in object_log_readers.read_object_log(log_path):
        states[entry.id] = entry.object

    return states


def _load_training_history_database(
    log_dir: str,
) -> object_log_readers.TrainingHistoryDatabase:
    return object_log_readers.TrainingHistoryDatabase(dirname=log_dir)


_LOADERS = {
    STATES_BY_STATE_ID_KEY: _load_states_by_state_id,
    ACTION_INVERSION_REPORTS_BY_BATCH_ID_KEY: _load_action_inversion_reports_by_batch_id,
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
