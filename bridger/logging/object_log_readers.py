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

# TODO(lyric): Consider adding batch_idx_min and batch_idx_max
# parameters to the data retrieval functions when the data volume
# grows enough such that debugging would benefit from a defined
# tighter window.
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
        self._q_values.set_index('state_id')
        self._td_errors = pd.DataFrame(read_object_log(dirname, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY))
        self._td_errors.set_index('state_id')


    def get_states_by_visit_count(self, n: int = None) -> pd.DataFrame:
        """Retrieves the top-n states by visit count.

        Args:
          n: The number of states to return.

        Returns:
          The top-n states sorted descending by visit count. The
          corresponding id and visit count are also included. The
          columns of the dataframe are state_id, state, and 
          visit_count.

        """        
        return self._visits.groupby(['object' ], sort=False)['batch_idx'].count().reset_index(name='visit_count').sort_values(['visit_count'], ascending=False).head(n).set_index('object').join(self._states).rename(columns={'object' : 'state', 'id' : 'state_id'})

    def get_td_errors(state_id: int, sample_count: int) -> pd.DataFrame:
        """Retrieves a sample of the td_error values for the requested state.

        Args:
          state_id: The state id for which to retrieve td errors across all actions.
          sample_count: The max number of td error values to return
            per action, uniformly spaced amongst matching values indepedent
            of batch_idx.

        """
        return 

# rest are given state, get t, q, target

# simple tests. port panel!


        
                

        
