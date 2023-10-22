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
import math
import numpy as np
import os
import pathlib
import pickle
import shutil
import torch

from collections.abc import Hashable
from typing import Optional, TypeVar, Generic
import torch

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


MetricMapValue = TypeVar("MetricMapValue", float, dict[int, int])


class MetricMap(Generic[MetricMapValue]):
    """Stores batch_idx and metric values for efficient access.

    For efficiency, the map operates in 'add' mode until it is
    'finalize'-d. It then becomes immutable.

    """

    def __init__(self):
        self._batch_idxs = []
        self._metric_values = []
        self._map = {}
        self._finalized = False

    def add(self, batch_idx: int, metric_value: MetricMapValue) -> None:
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
        # before logging. If the metric value being used is not a float,
        # an error will be thrown here if a duplicate value is found.

        # Keep the first value of the duplicate in the batch.

        duplicate_value = self._map.get(batch_idx)
        if duplicate_value:
            if not math.isclose(duplicate_value, metric_value, abs_tol=1e-5):
                raise ValueError(
                    "Metric values don't match for batch_idx duplicate. Current "
                    f"is {duplicate_value} and received {metric_value}."
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
        self._finalized = True

    def get(
        self, start_batch_idx: Optional[int], end_batch_idx: Optional[int]
    ) -> tuple[list[int], list[MetricMapValue]]:
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
        assert self._finalized
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


class StateActionMetricMap:
    """Stores metric values for efficient access.

    The optimized access pattern requests a series of batch_idx and
    metric values for a given state_id and action pair.

    For efficiency, the map operates in 'add' mode until it is
    'finalize'-d. It then becomes immutable.

    Attributes:
      nA: The likely number of actions based on the added data.
    """

    def __init__(self):
        self._map = {}
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
        self._map.setdefault(state_id, {}).setdefault(action, MetricMap[float]()).add(
            batch_idx=batch_idx, metric_value=metric_value
        )

    def get(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
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

        state_map = self._map.get(state_id)
        if state_map:
            entry = state_map.get(action)
            if entry:
                return entry.get(
                    start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
                )

        return [], []

    @property
    def nA(self) -> int:
        """Estimates the number of actions in the environment.

        The maximum action observed in the added log data is used as a
        proxy for the maximum action in the environment. Actions are
        consecutive ints so the (maximum value + 1) is the number of
        actions.

        Returns:
          The number of actions in the environment.

        """
        assert self._finalized
        return max([max(entry) for entry in self._map.values()]) + 1


@dataclasses.dataclass
class VisitEntry:
    """Describes a state and how often it was visited.

    Attributes:
      state_id: The id of the state described.
      state: The full state tensor.
      visit_count: The number of times the state was visited.

    """

    state_id: int
    state: torch.Tensor
    visit_count: int


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
      nA: The number of actions. Knowing the number of actions allows
        a user of TrainingHistoryDatabase to iteratively query metrics
        for all possible actions.

    """

    def __init__(self, dirname: str):
        """Loads in training history data for querying.

        Args:
          dirname: The directory containing the training history log files.
        """
        self._states = {}
        # State normalized log entries are stored in the parent of the object logging directory.
        for entry in _read_object_log(
            os.path.dirname(dirname), log_entry.STATE_NORMALIZED_LOG_ENTRY
        ):
            self._states[entry.id] = entry.object

        # Store visited states sorted by visit count.
        visit_counts = collections.defaultdict(int)
        for entry in _read_object_log(
            dirname, log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY
        ):
            visit_counts[entry.object] += 1
        sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        self._state_visits = []
        for visit in sorted_visits:
            state_id = visit[0]
            visit_count = visit[1]
            self._state_visits.append(
                VisitEntry(
                    state_id=state_id,
                    state=self._states[state_id],
                    visit_count=visit_count,
                )
            )

        # self._q_values = StateActionMetricMap()
        # self._q_target_values = StateActionMetricMap()
        # for entry in _read_object_log(
        #     dirname, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY
        # ):
        #     self._q_values.add(
        #         state_id=entry.state_id,
        #         action=entry.action,
        #         batch_idx=entry.batch_idx,
        #         metric_value=entry.q_value,
        #     )
        #     self._q_target_values.add(
        #         state_id=entry.state_id,
        #         action=entry.action,
        #         batch_idx=entry.batch_idx,
        #         metric_value=entry.q_target_value,
        #     )
        # self._q_values.finalize()
        # self._q_target_values.finalize()

        self._action_probability_values = StateActionMetricMap()
        for entry in _read_object_log(
            dirname, log_entry.TRAINING_HISTORY_ACTION_PROBABILITY_LOG_ENTRY
        ):
            self._action_probability_values.add(
                state_id=entry.state_id,
                action=entry.action,
                batch_idx=entry.batch_idx,
                metric_value=entry.action_probability,
            )
        self._action_probability_values.finalize()
        self.nA = self._action_probability_values.nA

        # self._td_errors = StateActionMetricMap()
        # for entry in _read_object_log(
        #     dirname, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY
        # ):
        #     self._td_errors.add(
        #         state_id=entry.state_id,
        #         action=entry.action,
        #         batch_idx=entry.batch_idx,
        #         metric_value=entry.td_error,
        #     )
        # self._td_errors.finalize()

        # self._replay_buffer_state_counts = MetricMap[dict[int, int]]()
        # for entry in _read_object_log(dirname, log_entry.TRAINING_BATCH_LOG_ENTRY):
        #     self._replay_buffer_state_counts.add(
        #         batch_idx=entry.batch_idx,
        #         metric_value={
        #             id: count for id, count in entry.replay_buffer_state_counts
        #         },
        #     )

        # self._replay_buffer_state_counts.finalize()

        # self.nA = self._td_errors.nA

    def get_states_by_visit_count(
        self,
        n: Optional[int] = None,
    ) -> list[VisitEntry]:
        """Retrieves the top-n states by visit count.

        Args:
          n: The number of states to return.

        Returns:
          The top-n states sorted descending by visit count. The
          corresponding state id and visit count are also
          included.

        """
        return self._state_visits[:n]

    def get_td_errors(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
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

    def get_action_probability_values(
        self,
        state_id: int,
        action: int,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
        """Retrieves state action probability values for the requested state and action.

        Args:
          state_id: The state id for which to retrieve action probability values.
          action: The action for which to retrieve action probability values.
          start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
          end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
          A tuple of lists of the same length. The first contains
          batch_idxs and the second contains corresponding action probability
          values.
        """
        return self._action_probability_values.get(
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
    ) -> tuple[list[int], list[float]]:
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
    ) -> tuple[list[int], list[float]]:
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

    def get_replay_buffer_state_counts(
        self,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> tuple[list[int], list[dict[int, int]]]:
        """Retrieves replay buffer state counts for the requested interval of start and end batch idxs.

        Args:
            start_batch_idx: The first batch index (inclusive) to consider
            when filtering the data.
            end_batch_idx: The last batch index (inclusive) to consider
            when filtering the data.

        Returns:
            A tuple of lists of the same length. The first contains
            batch_idxs and the second contains corresponding replay buffer state counts.

        """
        return self._replay_buffer_state_counts.get(
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
        for entry in _read_object_log(
            os.path.dirname(dirname), log_entry.STATE_NORMALIZED_LOG_ENTRY
        ):
            self._states[entry.id] = entry.object

        self._divergences = self._summarize_divergences()

    def _summarize_divergences(self) -> list[DivergenceEntry]:
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
        for batch_idx, reports in self._reports.items():
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
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> list[DivergenceEntry]:
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
        self,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
    ) -> tuple[list[int], list[int]]:
        """Returns the action inversion incident rate per batch.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A pair of parallel lists containing the batch_idx values and the
          corresponding action inversion incident rate.
        """
        batch_idxs = []
        incidence_rate = []
        for batch_idx, reports in self._reports.items():
            if start_batch_idx is not None and batch_idx < start_batch_idx:
                continue
            if end_batch_idx is not None and batch_idx > end_batch_idx:
                break

            batch_idxs.append(batch_idx)
            incidence_rate.append(len(reports))

        return batch_idxs, incidence_rate

    def get_reports(
        self, batch_idx: int
    ) -> list[tuple[log_entry.ActionInversionReportEntry, torch.Tensor]]:
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
