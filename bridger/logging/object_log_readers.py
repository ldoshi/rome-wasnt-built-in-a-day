"""Object log consumption tools to support various use-cases.

Supported use-cases range from simple and direct reading of a single
object log to the TrainingHistoryDatabase, which supports queries on
top of logged data.

"""
import collections
import copy
import dataclasses
import numpy as np
import os
import pathlib
import pandas as pd
import pickle
import shutil
import torch

from collections.abc import Hashable
from typing import List, Optional, Tuple

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
    df: pd.DataFrame,
    state_id: int,
    action: int,
    start_batch_idx: Optional[int],
    end_batch_idx: Optional[int],
    second_column: str,
) -> pd.DataFrame:
    """Retrieves batch_idx and second_column from df.

    Args:
      df: The dataframe from which to select rows.
      state_id: The state id for which to retrieve rows.
      action: The action for which to retrieve rows.
      start_batch_idx: The first batch index (inclusive) to consider
        when filtering df.
      end_batch_idx: The last batch index (inclusive) to consider when
        filtering df.
      second_column: The name of the second column to retrieve from df.

    Returns:
      A dataframe with two columns, batch_idx and second_column,
        filtered for rows that contain state_id and action as values
        in the corresponding columns.

    """
    df_filtered = df
    if start_batch_idx is not None:
        df_filtered = df_filtered[(df_filtered["batch_idx"] >= start_batch_idx)]
    if end_batch_idx is not None:
        df_filtered = df_filtered[(df_filtered["batch_idx"] <= end_batch_idx)]

    return df_filtered[
        (df_filtered["state_id"] == state_id) & (df_filtered["action"] == action)
    ][["batch_idx", second_column]]


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

    def get_states_by_visit_count(
        self,
        n: Optional[int] = None,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieves the top-n states by visit count.

        Args:
          n: The number of states to return.
          start_batch_idx: The first batch index (inclusive) to consider
            when computing the states by visit count.
          end_batch_idx: The last batch index (inclusive) to consider
            when computing the states by visit count.

        Returns:
          The top-n states sorted descending by visit count. The
          corresponding id and visit count are also included. The
          columns of the dataframe are state_id, state, and
          visit_count.

        """
        visits = self._visits
        if start_batch_idx is not None:
            visits = visits[(visits["batch_idx"] >= start_batch_idx)]
        if end_batch_idx is not None:
            visits = visits[(visits["batch_idx"] <= end_batch_idx)]

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

    def get_td_errors(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> pd.DataFrame:

        """Retrieves td_error values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve td errors.
          action: The action for which to retrieve td errors.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and td_error,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(
            df=self._td_errors,
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
            second_column="td_error",
        )

    def get_q_values(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> pd.DataFrame:

        """Retrieves q values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q values.
          action: The action for which to retrieve q values.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and q_values,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(
            df=self._q_values,
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
            second_column="q_value",
        )

    def get_q_target_values(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> pd.DataFrame:

        """Retrieves q target values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q target values.
          action: The action for which to retrieve q target values.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtered the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A dataframe with two columns, batch_idx and q_target_values,
            filtered for rows that contain state_id and action as values
            in the corresponding columns.
        """
        return _get_values_by_state_and_action(
            df=self._q_values,
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
            second_column="q_target_value",
        )


@dataclasses.dataclass
class DivergenceEntry:
    """Describes an incidence of divergence.

    Attributes:
      batch_idx: The batch_idx where the divergence occurs.
      convergence_run_length: The number of batches with 0 action
        inversion reports before this diversion occurred.
      divergence_magnitude: The number of action inversion reports
        logged in the moment of divergence.

    """

    batch_idx: int
    convergence_run_length: int
    divergence_magnitude: int


class ActionInversionDatabase:
    """Provides aggregate query access over logged action inversions.

    The ActionInversionDatabase combines the following log entries:
    * log_entry.ACTION_INVERSION_REPORT_ENTRY
    * log_entry.STATE_NORMALIZED_LOG_ENTRY

    """

    def __init__(self, dirname: str):
        """Loads in action inversion data into data structures for querying.

        Args:
          dirname: The directory containing the action inversion log files.
        """
        self._reports = collections.defaultdict(list)
        for entry in _read_object_log(dirname, log_entry.ACTION_INVERSION_REPORT_ENTRY):
            self._reports[entry.batch_idx].append(entry)

        self._states = {}
        for entry in _read_object_log(dirname, log_entry.STATE_NORMALIZED_LOG_ENTRY):
            self._states[entry.id] = entry.object

        self._divergences = self._summarize_divergences()

    def _summarize_divergences(self) -> List[DivergenceEntry]:
        """Summarizes incidents of divergence.

        A divergence is defined as an incident of logging at least one
        action inversion report following a period of one or more
        batches during which 0 action inversion reports logged.

        This function assumes ActionInversionReportEntry's are
        iterated in the same order in which they were logged, i.e. in
        increasing order of batch_idx.

        Returns:
          A list of DivergenceEntry to summarize all the divergences.

        """
        # TODO(lyric): Consider a more efficient data structure if
        # we're still using this tool with much larger log files.
        divergences = []
        last_batch_with_reports = 0
        for (batch_idx, reports) in self._reports.items():
            if len(reports):
                convergence_run_length = batch_idx - last_batch_with_reports - 1
                if convergence_run_length:
                    divergences.append(
                        DivergenceEntry(
                            batch_idx=batch_idx,
                            convergence_run_length=convergence_run_length,
                            divergence_magnitude=len(reports),
                        )
                    )
                last_batch_with_reports = batch_idx

        return divergences

    def get_divergences(
        self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None
    ) -> List[DivergenceEntry]:
        """Returns divergences occurring within the provided range.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A list of DivergenceEntry to summarize all the divergences
          between start_batch_idx and end_batch_idx.

        """
        start = 0
        if start_batch_idx is not None:
            for i, entry in enumerate(self._divergences):
                if start_batch_idx <= entry.batch_idx:
                    break
                start = i + 1

        end = len(self._divergences)
        if end_batch_idx is not None:
            for i in range(len(self._divergences) - 1, -1, -1):
                if self._divergences[i].batch_idx <= end_batch_idx:
                    break
                end = i

        return self._divergences[start:end]

    def get_incidence_rate(
        self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """Returns the action inversion incident rate per batch.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A pair of parallel lists containing the batch_idx values and the
          corresponding action inversion incdient rate.
        """
        batch_idxs = []
        incidence_rate = []
        for (batch_idx, reports) in self._reports.items():
            if start_batch_idx is not None and batch_idx < start_batch_idx:
                continue
            if end_batch_idx is not None and batch_idx > end_batch_idx:
                break

            batch_idxs.append(batch_idx)
            incidence_rate.append(len(reports))

        return batch_idxs, incidence_rate

    def get_reports(
        self, batch_idx: int
    ) -> List[Tuple[log_entry.ActionInversionReportEntry, torch.Tensor]]:
        """Returns action inversion reports paired with their corresponding states.

        Args:
          batch_idx: The index for which to return reports.

        Returns:
          A list containing pairs of the action inversion report and
          the corresponding full state.
        """
        if batch_idx not in self._reports:
            return []

        return [
            (report, self._states[report.state_id])
            for report in self._reports[batch_idx]
        ]
