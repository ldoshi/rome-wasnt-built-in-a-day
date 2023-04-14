"""Cache to load and store objects read in from logs.

The implementation relies on the fact that Flask in development mode
is single-process. Combined with the GIL, that means we do not expect
any race conditions to get/load data.

"""

import collections
import functools
import multiprocessing
import os
import time
import torch

from typing import Any, Callable, Dict, List, Optional, Tuple

from bridger.logging_utils import log_entry
from bridger.logging_utils import object_log_readers

ACTION_INVERSION_DATABASE_KEY = "action_inversion_database_key"
TRAINING_HISTORY_DATABASE_KEY = "training_history_database_key"


def get_experiment_data_dir(log_dir: str, experiment_name: str) -> str:
    return os.path.join(log_dir, experiment_name)


def _make_cache_key(experiment_name: str, data_key: str) -> Tuple[str, str]:
    return (experiment_name, data_key)


def _load_action_inversion_database(
    log_dir: str,
    experiment_name: str,
) -> Tuple[Optional[object_log_readers.ActionInversionDatabase], str]:
    if log_entry.ACTION_INVERSION_REPORT_ENTRY not in os.listdir(
        os.path.join(log_dir, experiment_name)
    ):
        return None, experiment_name

    return experiment_name, object_log_readers.ActionInversionDatabase(
        dirname=get_experiment_data_dir(
            log_dir=log_dir, experiment_name=experiment_name
        )
    )


def _load_training_history_database(
    log_dir: str,
    experiment_name: str,
) -> Tuple[str, object_log_readers.TrainingHistoryDatabase]:
    return experiment_name, object_log_readers.TrainingHistoryDatabase(
        dirname=get_experiment_data_dir(
            log_dir=log_dir, experiment_name=experiment_name
        )
    )


_LOADERS = {
    ACTION_INVERSION_DATABASE_KEY: _load_action_inversion_database,
    TRAINING_HISTORY_DATABASE_KEY: _load_training_history_database,
}


class ObjectLogCache:
    """Loads and caches logged objects for efficient re-access.

    Attributes:
      hit_counts: Dict of the number of cache hits per key.
      miss_counts: Dict of the number of cache misses per key.
    """

    def __init__(
        self, log_dir: str, loaders: Dict[str, Callable[[str, str], Any]] = _LOADERS
    ):
        """Initializes cache.

        Args:
          log_dir: The base directory containing the object logging
            subdirectories and log files.
          loaders: A dict containing a custom load function for each
            supported data_key. On cache miss, the load function will
            be called and its return value will be cached as the value
            corresponding to the key.

        """

        self._log_dir = log_dir
        self._cache = {}
        self._loaders = loaders
        self.hit_counts = collections.defaultdict(int)
        self.miss_counts = collections.defaultdict(int)

    def warm(self, experiment_names: List[str]) -> None:
        """Warms the cache by loading all data keys for all experiments.

        All data loading is done using multiprocessing.
        """
        for data_key in self._loaders.keys():
            loader = functools.partial(self._loaders[data_key], self._log_dir)

            # Chose only 2 background processes since most of the
            # processing time is related to reading data from disk. If
            # I understand correctly, multiple processes don't speed
            # this up too much unless there's corresponding hardware
            # support with multiple disk heds.
            with multiprocessing.Pool(processes=2) as pool:
                for experiment_name, data in pool.imap_unordered(
                    loader, experiment_names
                ):
                    if data is None:
                        continue
                    cache_key = _make_cache_key(experiment_name, data_key)
                    if cache_key not in self._cache:
                        self._cache[cache_key] = data

    def get(self, experiment_name: str, data_key: str) -> Any:
        """Retrieves the requested data constructed from log entries.

        If the data has not been loaded yet, it will be loaded here
        first. Each support date_key type has a custom log entry loader.

        Args:
          experiment_name: The name of the experiment of interest.
          data_key: The key describing the desired log-based data.

        Returns:
          Logged data loaded into a data structure corresponding to
          the data_key for the requested experiment. See loader
          definitions. None if backing files do not exist at the
          provided location.

        Raises:
          ValueError on unsupported data_key.
          FileNotFoundError if the (experiment_name, data_key) pair
            does not correspond to a backing data file.

        """
        if data_key not in self._loaders:
            raise ValueError(f"Unsupported data_key: {data_key}")

        cache_key = _make_cache_key(experiment_name, data_key)
        if cache_key not in self._cache:
            self.miss_counts[cache_key] += 1
            start = int(time.time() * 1e3)
            _, self._cache[cache_key] = self._loaders[data_key](
                self._log_dir, experiment_name
            )
            end = int(time.time() * 1e3)
            print(
                f"ObjectLogCache loading {(experiment_name, data_key)} "
                f"took {end-start} ms."
            )
        else:
            self.hit_counts[cache_key] += 1

        return self._cache[cache_key]
