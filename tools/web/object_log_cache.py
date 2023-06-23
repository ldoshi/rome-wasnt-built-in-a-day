"""Cache to load and store objects read in from logs.

The implementation relies on the fact that Flask in development mode
is single-process. Combined with the GIL, that means we do not expect
any race conditions to get/load data.

"""

import collections
import functools
import multiprocessing
import os
import pickle
import time
import torch

from typing import Any, Callable, Dict, List

from bridger.logging_utils import log_entry
from bridger.logging_utils import object_log_readers

ACTION_INVERSION_DATABASE_KEY = "action_inversion_database_key"
TRAINING_HISTORY_DATABASE_KEY = "training_history_database_key"

DatabaseType = (
    object_log_readers.TrainingHistoryDatabase
    | object_log_readers.ActionInversionDatabase
)


def get_experiment_data_dir(log_dir: str, experiment_name: str) -> str:
    return os.path.join(log_dir, experiment_name)


def _make_cache_key(experiment_name: str, data_key: str) -> tuple[str, str]:
    return (experiment_name, data_key)


def _make_subdir_if_necessary(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_action_inversion_database_from_log(
    log_dir: str,
    experiment_name: str,
) -> object_log_readers.ActionInversionDatabase:
    return object_log_readers.ActionInversionDatabase(
        dirname=get_experiment_data_dir(
            log_dir=log_dir, experiment_name=experiment_name
        )
    )


def _load_training_history_database_from_log(
    log_dir: str,
    experiment_name: str,
) -> object_log_readers.TrainingHistoryDatabase:
    return object_log_readers.TrainingHistoryDatabase(
        dirname=get_experiment_data_dir(
            log_dir=log_dir, experiment_name=experiment_name
        )
    )


def _save_database(
    directory: str, experiment_name: str, database: DatabaseType
) -> None:
    with open(os.path.join(directory, experiment_name), "wb") as f:
        pickle.dump(database, f)


def _database_exists(directory: str, experiment_name: str) -> bool:
    return os.path.isfile(os.path.join(directory, experiment_name))


def _load_database(directory: str, experiment_name: str) -> DatabaseType:
    with open(os.path.join(directory, experiment_name), "rb") as f:
        return pickle.load(f)

def _convert_log_to_saved_database_if_necessary(loader_fn: Callable[[str], DatabaseType], database_directory: str, experiment_name: str) -> None:
    print("CALLED WITH " , database_directory, "experiment_name " , experiment_name)
    if _database_exists(directory=database_directory, experiment_name=experiment_name):
        return
    database = loader_fn(experiment_name)
    _save_database(directory=database_directory, experiment_name=experiment_name,database=database)
                                                                     

_LOADERS = {
    ACTION_INVERSION_DATABASE_KEY: _load_action_inversion_database_from_log,
    TRAINING_HISTORY_DATABASE_KEY: _load_training_history_database_from_log,
}


class ObjectLogCache:
    """Loads and caches logged objects for efficient re-access.

    Attributes:
      hit_counts: Dict of the number of cache hits per key.
      miss_counts: Dict of the number of cache misses per key.
      load_database_hit_counts: Dict of the number of load calls per
        key that loaded the database directly.
      load_database_miss_counts: Dict of the number of load calls per
        key that loaded the database from log.

    """

    def __init__(self, log_dir: str, temp_dir: str):
        """Initializes cache.

        Args:
          log_dir: The base directory containing the object logging
            subdirectories and log files.
          temp_dir: A directory for the ObjectLogCache to use for
            writing and reading temp files to increase experiment data
            load speed.

        """

        self._log_dir = log_dir
        self._temp_dir = temp_dir
        self._cache = {}
        self.hit_counts = collections.defaultdict(int)
        self.miss_counts = collections.defaultdict(int)
        self.load_database_hit_counts = collections.defaultdict(int)
        self.load_database_miss_counts = collections.defaultdict(int)

    def convert_logs_to_saved_databases(self, experiment_names: list[str]) -> None:
        """Converts log data into saved databases for all data keys and experiments.

        Loading saved databases into the ObjectLogCache is much faster than constructing them from logs. 

        All data loading is done using multiprocessing.

        Args:
          experiment_names: The names of all the experiments to convert.
        """
        for data_key, loader in _LOADERS.items():
            loader_fn = functools.partial(loader, self._log_dir)
            convert_log_to_saved_database_if_necessary = functools.partial(_convert_log_to_saved_database_if_necessary, loader_fn, self._temp_dir)

            # Chose only 2 background processes since most of the
            # processing time is related to reading data from disk. If
            # I understand correctly, multiple processes don't speed
            # this up too much unless there's corresponding hardware
            # support with multiple disk heads.
            with multiprocessing.Pool(processes=2) as pool:
                pool.map(convert_log_to_saved_database_if_necessary, experiment_names)
        
    def _load(self, experiment_name: str, data_key: str) -> DatabaseType:
        if data_key not in _LOADERS:
            raise ValueError(f"Unsupported data_key: {data_key}")

        cache_key = _make_cache_key(experiment_name=experiment_name, data_key=data_key)

        temp_dir_path = os.path.join(self._temp_dir, data_key)
        if _database_exists(directory=temp_dir_path, experiment_name=experiment_name):
            self.load_database_hit_counts[cache_key] += 1
            return _load_database(
                directory=temp_dir_path, experiment_name=experiment_name
            )

        _make_subdir_if_necessary(temp_dir_path)
        database = _LOADERS[data_key](
            log_dir=self._log_dir, experiment_name=experiment_name
        )
        self.load_database_miss_counts[cache_key] += 1
        _save_database(
            directory=temp_dir_path, experiment_name=experiment_name, database=database
        )
        return database

    def get(self, experiment_name: str, data_key: str) -> Any:
        """Retrieves the requested data constructed from log entries.

        If the data has not been loaded yet, it will be loaded here
        first.

        Args:
          experiment_name: The name of the experiment of interest.
          data_key: The key describing the desired log-based data.

        Returns:
          Logged data loaded into a data structure corresponding to
          the data_key for the requested experiment. None if backing
          files do not exist at the provided location.

        Raises:
          ValueError on unsupported data_key.
          FileNotFoundError if the (experiment_name, data_key) pair
            does not correspond to a backing data file.

        """

        cache_key = _make_cache_key(experiment_name=experiment_name, data_key=data_key)
        if cache_key not in self._cache:
            self.miss_counts[cache_key] += 1
            start = int(time.time() * 1e3)
            self._cache[cache_key] = self._load(
                experiment_name=experiment_name, data_key=data_key
            )
            end = int(time.time() * 1e3)
            print(
                f"ObjectLogCache loading {(experiment_name, data_key)} "
                f"took {end-start} ms."
            )
        else:
            self.hit_counts[cache_key] += 1

        return self._cache[cache_key]
