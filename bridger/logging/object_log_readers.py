"""Object log consumption tools to support various use-cases.

Supported use-cases range from simple and direct reading of a single
object log to the TrainingHistoryDatabase, which supports queries on
top of logged data.

"""
import collections
import copy
import numpy as np
import os
import pathlib
import pandas as pd
import pickle
import shutil
import torch

from collections.abc import Hashable
from typing import Optional

from bridger.logging import log_entry


def read_object_log(log_filepath: str):
    with open(log_filepath, "rb") as f:
        buffer = None
        while True:
            try:
                buffer = pickle.load(f)

                for element in buffer:
                    yield element

            except EOFError:
                break


def _read_object_log(dirname: str, log_filename: str):
    yield from read_object_log(log_filepath=os.path.join(dirname, log_filename))


def _get_values_by_state_and_action(
        df: pd.DataFrame, state_id: int, action: int, start_batch_index: Optional[int], end_batch_index: Optional[int], second_column: str
) -> pd.DataFrame:
    """Retrieves batch_idx and second_column from df.

    Args:
      df: The dataframe from which to select rows.
      state_id: The state id for which to retrieve rows.
      action: The action for which to retrieve rows.
      start_batch_index: The first batch index (inclusive) to
        consider when filtering df.
      end_batch_index: The last batch index (inclusive) to
        consider when filtering df.
      second_column: The name of the second column to retrieve from df.

    Returns:
      A dataframe with two columns, batch_idx and second_column,
        filtered for rows that contain state_id and action as values
        in the corresponding columns.
    """
    df_filtered = df
    if start_batch_index is not None:
        df_filtered = df_filtered[(df_filtered['batch_idx'] >= start_batch_index)]
    if end_batch_index is not None:
        df_filtered = df_filtered[(df_filtered['batch_idx'] <= end_batch_index)]
    
    return df_filtered[(df_filtered["state_id"] == state_id) & (df_filtered["action"] == action)][
        ["batch_idx", second_column]
    ]


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

    Attributes:
      actions_n: The value of the max action found in the logs.
    """

    def __init__(self, dirname: str):
        """Loads in training history data in dataframes for querying.

        Args:
          dirname: The directory containing the training history log files.
        """
        self._states = pd.DataFrame(
            _read_object_log(dirname, log_entry.STATE_NORMALIZED_LOG_ENTRY)
        )
        self._states.set_index("id")

        self._visits = pd.DataFrame(
            _read_object_log(dirname, log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY)
        )

        self._q_values = pd.DataFrame(
            _read_object_log(dirname, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY)
        )
        self._q_values = self._q_values.sort_values(
            by=["state_id", "action", "batch_idx"]
        )
        self._q_values.set_index("state_id")

        self._td_errors = pd.DataFrame(
            _read_object_log(dirname, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY)
        )
        self._td_errors = self._td_errors.drop_duplicates(
            ["state_id", "action", "batch_idx"]
        ).sort_values(by=["state_id", "action", "batch_idx"])
        self._td_errors.set_index("state_id")

        self.actions_n = (
            max(self._q_values["action"].max(), self._td_errors["action"].max()) + 1
        )

    def get_states_by_visit_count(self, n: Optional[int] = None, start_batch_index: Optional[int] = None, end_batch_index: Optional[int] = None) -> pd.DataFrame:
        """Retrieves the top-n states by visit count.

        Args:
          n: The number of states to return.
          start_batch_index: The first batch index (inclusive) to
            consider when computing the states by visit count.
          end_batch_index: The last batch index (inclusive) to
            consider when computing the states by visit count.

        Returns:
          The top-n states sorted descending by visit count. The
          corresponding id and visit count are also included. The
          columns of the dataframe are state_id, state, and
          visit_count.

        """
        visits = self._visits
        if start_batch_index is not None:
            visits = visits[(visits['batch_idx'] >= start_batch_index)]
        if end_batch_index is not None:
            visits = visits[(visits['batch_idx'] <= end_batch_index)]

        return (
            visits.groupby(["object"], sort=False)["batch_idx"]
            .count()
            .reset_index(name="visit_count")
            .sort_values(["visit_count"], ascending=False)
            .head(n)
            .set_index("object")
            .join(self._states)
            .rename(columns={"object": "state", "id": "state_id"})
        )

    def get_td_errors(self, state_id: int, action: int, start_batch_index: Optional[int] = None, end_batch_index: Optional[int] = None) -> pd.DataFrame:

        """Retrieves td_error values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve td errors.
          action: The action for which to retrieve td errors.
          start_batch_index: The first batch index (inclusive) to
            consider when filtering the data.
          end_batch_index: The last batch index (inclusive) to
            consider when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and td_error,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(
            df=self._td_errors, state_id=state_id, action=action, start_batch_index=start_batch_index, end_batch_index=end_batch_index, second_column="td_error"        )

    def get_q_values(self, state_id: int, action: int, start_batch_index: Optional[int] = None, end_batch_index: Optional[int] = None) -> pd.DataFrame:

        """Retrieves q values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q values.
          action: The action for which to retrieve q values.
          start_batch_index: The first batch index (inclusive) to
            consider when filtering the data.
          end_batch_index: The last batch index (inclusive) to
            consider when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and q_values,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(
            df=         self._q_values, state_id=state_id,action= action,start_batch_index=start_batch_index,end_batch_index=end_batch_index,second_column= "q_value"
        )

    def get_q_target_values(self, state_id: int, action: int, start_batch_index: Optional[int] = None, end_batch_index: Optional[int] = None) -> pd.DataFrame:

        """Retrieves q target values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q target values.
          action: The action for which to retrieve q target values.
          start_batch_index: The first batch index (inclusive) to
            consider when filtered the data.
          end_batch_index: The last batch index (inclusive) to
            consider when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and q_target_values,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(df=            self._q_values, state_id=state_id, action=action, start_batch_index=start_batch_index,end_batch_index=end_batch_index,second_column= "q_target_value"        )
