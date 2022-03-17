"""Object log consumption tools to support various use-cases.

Supported use-cases range from simple and direct reading of a single
object log to the TrainingHistoryDatabase, which supports aggregate
queries on top of logged data.

"""
from typing import Any, Callable, List, Optional

import collections
import copy
import shutil
import pickle
import os
import pathlib
import pandas as pd
import torch

from collections.abc import Hashable

from bridger.logging import log_entry


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


class TrainingHistoryDatabase:
    """Provides aggregate query access over logged training history.

    The TrainingHistoryDatabase combines the following log entries:
    * log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY
    * log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY
    * log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY
    * log_entry.STATE_NORMALIZED_LOG_ENTRY
    """

    def __init__(self, dirname: str):
        """Loads in training history data in dataframes for querying.

        Args:
          dirname: The directory containing the training history log files.
        """
        self._states = pd.DataFrame(read_object_log(dirname, log_entry.STATE_NORMALIZED_LOG_ENTRY))
        self._states.set_index('id')
        self._visits = pd.DataFrame(read_object_log(dirname, log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY))
        self._q_values = pd.DataFrame(read_object_log(dirname, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY))
        self._td_errors = pd.DataFrame(read_object_log(dirname, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY))


    def get_states_by_visit_count(n: int = None) -> pd.DataFrame:
        return self._visits.groupby(['object' ], sort=False)['batch_idx'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(n).join(self._states, rsuffix='_state').rename(columns={'object_state' : 'state'}).drop(['object'], axis=1)

    def get_q_values(state_ids: pd.DataFrame) -> 

    # potentially add sampling? 
# rest are given states, get t, q, target
        
                

        
