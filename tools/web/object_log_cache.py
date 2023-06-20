"""Cache to load and store objects read in from logs.

The implementation relies on the fact that Flask in development mode
is single-process. Combined with the GIL, that means we do not expect
any race conditions to get/load data.

"""

import collections
import os
import pickle
import time
import torch

from typing import Any, Callable, Dict, List, Union

from bridger.logging_utils import log_entry
from bridger.logging_utils import object_log_readers

ACTION_INVERSION_DATABASE_KEY = "action_inversion_database_key"
TRAINING_HISTORY_DATABASE_KEY = "training_history_database_key"

def get_experiment_data_dir(log_dir: str, experiment_name: str) -> str:
    return os.path.join(log_dir, experiment_name)


def _make_temp_subdir_if_necessary(path: str) -> None:
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


def _save_database(directory: str, experiment_name: str, database: Union[object_log_readers.TrainingHistoryDatabase|object_log_readers.TrainingHistoryDatabase]) -> None:
    with open(os.path.join(directory, experiment_name), "wb") as f:
        pickle.dump(database, f)
    

def _database_exists(directory: str, experiment_name: str) -> bool:
    return os.path.isfile(os.path.join(directory, experiment_name))

def _load_database(directory: str, experiment_name: str) -> Union[object_log_readers.TrainingHistoryDatabase|object_log_readers.TrainingHistoryDatabase]:
    with open(os.path.join(directory, experiment_name), 'rb') as f:
        return pickle.load(f)

    
_LOADERS = {
    ACTION_INVERSION_DATABASE_KEY: _load_action_inversion_database_from_log,
    TRAINING_HISTORY_DATABASE_KEY: _load_training_history_database_from_log,
}



# TODO(lyric): Loaded from log stats. unit test include create two object log caches. move the helpers into the class as members.

class ObjectLogCache:
    """Loads and caches logged objects for efficient re-access.

    Attributes:
      hit_counts: Dict of the number of cache hits per key.
      miss_counts: Dict of the number of cache misses per key.    
      loaded_from_log: Dict describing whether a key was loaded from log. 
    """

    def __init__(
            self, log_dir: str, temp_dir: str    ):
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
        self.loaded_from_log = {}


    def _load(
            self,
        experiment_name: str,
        data_key: str
    ) -> Union[object_log_readers.TrainingHistoryDatabase|object_log_readers.TrainingHistoryDatabase]:
        if data_key not in _LOADERS:
            raise ValueError(f"Unsupported data_key: {data_key}")

        temp_dir_path = os.path.join(self._temp_dir, data_key)
        if _database_exists(directory=temp_dir_path, experiment_name=experiment_name):
            return _load_database(directory=temp_dir_path, experiment_name=experiment_name)

        _make_temp_subdir_if_necessary(temp_dir_path)
        database = _LOADERS[data_key](                log_dir=self._log_dir, experiment_name=experiment_name            )
        _save_database(directory=temp_dir_path, experiment_name=experiment_name, database=database)
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

        cache_key = (experiment_name, data_key)
        if cache_key not in self._cache:
            self.miss_counts[cache_key] += 1
            start = int(time.time() * 1e3)
            self._cache[cache_key] = self._load(                                                            experiment_name=experiment_name, data_key=data_key)
            end = int(time.time() * 1e3)
            print(
                f"ObjectLogCache loading {(experiment_name, data_key)} "
                f"took {end-start} ms."
            )
        else:
            self.hit_counts[cache_key] += 1

        return self._cache[cache_key]
