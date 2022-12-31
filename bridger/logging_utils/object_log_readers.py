"""Object log consumption tools to support various use-cases.

Supported use-cases range from simple and direct reading of a single
object log to the TrainingHistoryDatabase, which supports queries on
top of logged data.

"""
import bisect
import collections
import copy
import dataclasses
import itertools
import numpy as np
import os
import pathlib
import pandas as pd
import pickle
import shutil
import torch

from collections.abc import Hashable
from typing import List, Optional, Tuple

from bridger.logging_utils import log_entry


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


class _MetricMapEntry:
    """Stores batch_idx and metric values for efficient access.

    For efficiency, the map operates in 'add' mode until it is
    'finalize'-d. It then becomes immutable.

    """

    def __init__(self):
        self._batch_idxs = []
        self._metric_values = []
        self._map = {}
        self._finalized = False

    def add(self, batch_idx: int, metric_value: float) -> None:
        """Adds a batch_idx and metric_value pair.

        Repeated calls to add must provide the batch_idxs in
        increasing order. This is asserted once during finalize for
        efficiency. Providing a duplicate (batch_idx, metric_value) is
        permitted and will be ignored. Providing different
        metric_values for the same batch_idx is an error.

        Args:
          batch_idx: A batch idx.
          metric_value: The metric value corresponding to the batch idx.

        Raises:
          ValueError: If a batch_idx is provided with a different
            metric_value than before.

        """
        assert not self._finalized

        # Don't re-add duplicates. Consider removing this check and
        # presuming the data satisfied this invariant when it was
        # logged. Note that this invariant is not currently checked
        # before logging.
        duplicate_value = self._map.get(batch_idx)
        if duplicate_value:
            if duplicate_value != metric_value:
                raise ValueError(
                    "Metric values don't match for batch_idx duplicate. Current "
                    f"is {duplicate_value} and received {metric_value}"
                )
            return

        self._map[batch_idx] = metric_value

    def finalize(self):
        """Converts from add-optimized to get-optimized mode."""
        assert not self._finalized
        self._batch_idxs = list(self._map)
        self._metric_values = list(self._map.values())

        assert self._batch_idxs == sorted(
            self._batch_idxs
        ), "Batch_idxs must be added in increasing order."

        del self._map

    def get(
        self, start_batch_idx: Optional[int], end_batch_idx: Optional[int]
    ) -> Tuple[List[int], List[float]]:
        """Retrieves batch_idx and metric values for the requested range.

        Args:
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering metric values.
          end_batch_idx: The last batch index (inclusive) to consider when
            filtering metric values.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding metric
          values.
        """
        start_index = (
            0
            if start_batch_idx is None
            else bisect.bisect_left(self._batch_idxs, start_batch_idx)
        )
        end_index = (
            len(self._batch_idxs)
            if end_batch_idx is None
            else bisect.bisect_right(self._batch_idxs, end_batch_idx)
        )
        return (
            self._batch_idxs[start_index:end_index],
            self._metric_values[start_index:end_index],
        )


class MetricMap:
    """Stores metric values for efficient access.

    The optimized access pattern requests a series of batch_idx and
    metric values for a given state_id and action pair.

    For efficiency, the map operates in 'add' mode until it is
    'finalize'-d. It then becomes immutable.

    Attributes:
      nA: The likely number of actions based on the added data.
    """

    def __init__(self):
        self._map = collections.defaultdict(
            lambda: collections.defaultdict(lambda: _MetricMapEntry())
        )
        self._finalized = False

    def finalize(self):
        """Converts from add-optimized to get-optimized mode."""
        assert not self._finalized
        for actions_map in self._map.values():
            for metric_map_entry in actions_map.values():
                metric_map_entry.finalize()
        self._finalized = True

    def add(
        self, state_id: int, action: int, batch_idx: int, metric_value: float
    ) -> None:
        """Adds a batch_idx and metric_value organized by state_id and action.

        Repeated calls to add must provide the batch_idxs in
        increasing order for a given state_id and action. Providing a
        duplicate (batch_idx, metric_value) is permitted for a given
        state_id and action and will be ignored. Providing different
        metric_values for the same (state_id, action, batch_idx) is an
        error.

        Args:
          state_id: The state id corresponding to the metric_value.
          action: The action corresponding to the metric_value.
          batch_idx: The batch idx corresponding to the metric_value.
          metric_value: A metric value recorded for the (state_id,
            action, batch_idx) triple.

        Raises:
          ValueError: If the batch_idx provided is smaller than the
            most recent batch_idx provided for a given state_id and
            action pair. If a batch_idx is provided with a different
            metric_value than before for a given state_id and action
            pair.
        """
        assert not self._finalized
        self._map[state_id][action].add(batch_idx=batch_idx, metric_value=metric_value)

    def get(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Retrieves batch_idx and metric values for the requested range.

        Args:
          state_id: The state id for which to retrieve rows.
          action: The action for which to retrieve rows.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering metric values.
          end_batch_idx: The last batch index (inclusive) to consider when
            filtering metric values.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding metric
          values.
        """
        assert self._finalized
        return self._map[state_id][action].get(
            start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
        )

    @property
    def nA(self):
        return max([max(entry) for entry in self._map.values()]) + 1


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


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def _test(data):

    q_values = MetricMap()
    q_target_values = MetricMap()

    counter = 0
    for entry in data:
        if not entry:
            break
        counter += 1
        q_values.add(
            state_id=entry.state_id,
            action=entry.action,
            batch_idx=entry.batch_idx,
            metric_value=entry.q_value,
        )
        q_target_values.add(
            state_id=entry.state_id,
            action=entry.action,
            batch_idx=entry.batch_idx,
            metric_value=entry.q_target_value,
        )

    q_values.finalize()
    q_target_values.finalize()
    return q_values, q_target_values


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
      nA: The number of actions.

    """

    def __init__(self, dirname: str):
        """Loads in training history data in dataframes for querying.

        Args:
          dirname: The directory containing the training history log files.
        """
        import time

        start = int(time.time() * 1e3)
        self._q_values = MetricMap()
        self._q_target_values = MetricMap()
        for entry in _read_object_log(
            dirname, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY
        ):
            self._q_values.add(
                state_id=entry.state_id,
                action=entry.action,
                batch_idx=entry.batch_idx,
                metric_value=entry.q_value,
            )
            self._q_target_values.add(
                state_id=entry.state_id,
                action=entry.action,
                batch_idx=entry.batch_idx,
                metric_value=entry.q_target_value,
            )
        end = int(time.time() * 1e3)
        print(f"0 yoyoyo that took : {end-start})")
        self._q_values.finalize()
        self._q_target_values.finalize()
        end = int(time.time() * 1e3)
        print(f"1 yoyoyo that took : {end-start})")

        self._states = pd.DataFrame(
            _read_object_log(dirname, log_entry.STATE_NORMALIZED_LOG_ENTRY)
        )
        self._states.set_index("id")
        end = int(time.time() * 1e3)
        print(f"2 yoyoyo that took : {end-start})")

        self._visits = pd.DataFrame(
            _read_object_log(dirname, log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY)
        )
        end = int(time.time() * 1e3)
        print(f"3 yoyoyo that took : {end-start})")

        self._samples = pd.DataFrame(
            _read_object_log(dirname, log_entry.TRAINING_BATCH_LOG_ENTRY)
        )
        end = int(time.time() * 1e3)
        print(f"4 yoyoyo that took : {end-start})")

        # end = int(time.time() * 1e3)
        # print(f"yoyoyo that took : {end-start})")

        self._td_errors = MetricMap()
        for entry in _read_object_log(
            dirname, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY
        ):
            self._td_errors.add(
                state_id=entry.state_id,
                action=entry.action,
                batch_idx=entry.batch_idx,
                metric_value=entry.td_error,
            )
        self._td_errors.finalize()
        end = int(time.time() * 1e3)
        print(f"5 yoyoyo that took : {end-start})")

        self.actions_n = max(
            self._q_values.nA, self._q_target_values.nA, self._td_errors.nA
        )
        end = int(time.time() * 1e3)
        print(f"6 yoyoyo that took : {end-start})")

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
    ) -> Tuple[List[int], List[float]]:
        """Retrieves td_error values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve td errors.
          action: The action for which to retrieve td errors.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding td error
          values.
        """
        return self._td_errors.get(
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
        )

    def get_q_values(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Retrieves q values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q values.
          action: The action for which to retrieve q values.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding td error
          values.
        """
        return self._q_values.get(
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
        )

    def get_q_target_values(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Retrieves q target values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve q target values.
          action: The action for which to retrieve q target values.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtered the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding td error
          values.
        """
        return self._q_target_values.get(
            state_id=state_id,
            action=action,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
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
        self,
        start_batch_index: Optional[int] = None,
        end_batch_index: Optional[int] = None,
    ) -> List[DivergenceEntry]:
        """Returns divergences occurring within the provided range.

        Args:
          start_batch_index: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_index: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A list of DivergenceEntry to summarize all the divergences
          between start_batch_index and end_batch_index.

        """
        start = 0
        if start_batch_index is not None:
            for i, entry in enumerate(self._divergences):
                if start_batch_index <= entry.batch_idx:
                    break
                start = i + 1

        end = len(self._divergences)
        if end_batch_index is not None:
            for i in range(len(self._divergences) - 1, -1, -1):
                if self._divergences[i].batch_idx <= end_batch_index:
                    break
                end = i

        return self._divergences[start:end]

    def get_incidence_rate(
        self,
        start_batch_index: Optional[int] = None,
        end_batch_index: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Returns the action inversion incident rate per batch.

        Args:
          start_batch_index: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_index: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A pair of parallel lists containing the batch_idx values and the
          corresponding action inversion incident rate.
        """
        batch_idxs = []
        incidence_rate = []
        for (batch_idx, reports) in self._reports.items():
            if start_batch_index is not None and batch_idx < start_batch_index:
                continue
            if end_batch_index is not None and batch_idx > end_batch_index:
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
