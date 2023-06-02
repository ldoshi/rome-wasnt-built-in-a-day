import dataclasses
import pickle
import math
import itertools
import os
import unittest

import torch
from typing import Any, List, Optional

from parameterized import parameterized

from bridger.logging_utils import log_entry
from bridger.logging_utils import object_logging
from bridger.logging_utils import object_log_readers
from bridger import test_utils


_LOG_FILENAME_0 = "log_filename_0_pytest"


class TestStateActionMetricMap(unittest.TestCase):
    def test_add_smaller(self):
        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=10, metric_value=2.5)
        # Adding an entry with smaller batch_idx for a different
        # state_id or action is ok.
        test_map.add(state_id=0, action=1, batch_idx=9, metric_value=2.5)
        test_map.add(state_id=1, action=0, batch_idx=9, metric_value=2.5)
        test_map.finalize()

        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=10, metric_value=2.5)
        # Adding an entry with smaller batch_idx for the same state_id
        # and action is not ok.
        test_map.add(state_id=0, action=0, batch_idx=9, metric_value=2.5)
        self.assertRaisesRegex(
            AssertionError, "must be added in increasing order", test_map.finalize
        )

    def test_add_almost_duplicate(self):
        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=10, metric_value=2.5)
        self.assertRaisesRegex(
            ValueError,
            "Metric values don't match for batch_idx duplicate",
            test_map.add,
            state_id=0,
            action=0,
            batch_idx=10,
            metric_value=2.6,
        )

    def test_illegal_order_of_operations(self):
        test_map = object_log_readers.StateActionMetricMap()

        self.assertRaises(AssertionError, test_map.get, state_id=0, action=0)

        with self.assertRaises(AssertionError):
            test_map.nA

        test_map.finalize()

        self.assertRaises(
            AssertionError,
            test_map.add,
            state_id=0,
            action=0,
            batch_idx=0,
            metric_value=0,
        )

        self.assertRaises(AssertionError, test_map.finalize)

    def test_add_duplicate(self):
        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=10, metric_value=2.00000)
        test_map.add(state_id=0, action=0, batch_idx=10, metric_value=2.000009)
        test_map.finalize()

        batch_idxs, values = test_map.get(state_id=0, action=0)

        expected_batch_idxs = [10]
        expected_values = [2.000000]
        self.assertEqual(batch_idxs, expected_batch_idxs)
        self.assertEqual(values, expected_values)

    @parameterized.expand(
        [
            ("neither", None, None),
            ("no left", None, 3),
            ("no right", 2, None),
            ("both", 2, 3),
        ]
    )
    def test_empty_get_with_batch_idx_filters(
        self, name: str, start_batch_idx: Optional[int], end_batch_idx: Optional[int]
    ):
        test_map = object_log_readers.StateActionMetricMap()
        test_map.finalize()
        batch_idxs, values = test_map.get(
            state_id=0,
            action=0,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
        )
        self.assertFalse(batch_idxs)
        self.assertFalse(values)

    @parameterized.expand(
        [
            ("neither", None, None, [1, 2, 3, 4], [1.1, 2.1, 3.1, 4.1]),
            ("no left", None, 3, [1, 2, 3], [1.1, 2.1, 3.1]),
            ("no right", 2, None, [2, 3, 4], [2.1, 3.1, 4.1]),
            ("both", 2, 3, [2, 3], [2.1, 3.1]),
        ]
    )
    def test_get_with_batch_idx_filters(
        self,
        name: str,
        start_batch_idx: Optional[int],
        end_batch_idx: Optional[int],
        expected_batch_idxs: List[int],
        expected_values: List[float],
    ):
        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=1, metric_value=1.1)
        test_map.add(state_id=0, action=0, batch_idx=2, metric_value=2.1)
        test_map.add(state_id=0, action=0, batch_idx=3, metric_value=3.1)
        test_map.add(state_id=0, action=0, batch_idx=4, metric_value=4.1)
        test_map.finalize()

        batch_idxs, values = test_map.get(
            state_id=0,
            action=0,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
        )
        self.assertEqual(batch_idxs, expected_batch_idxs)
        self.assertEqual(values, expected_values)

    def test_get_different_states_and_actions(self):
        test_map = object_log_readers.StateActionMetricMap()

        test_map.add(state_id=0, action=0, batch_idx=1, metric_value=1.1)
        test_map.add(state_id=0, action=1, batch_idx=2, metric_value=2.1)
        test_map.add(state_id=1, action=0, batch_idx=1, metric_value=3.1)
        test_map.finalize()

        batch_idxs, values = test_map.get(state_id=0, action=0)
        self.assertEqual(batch_idxs, [1])
        self.assertEqual(values, [1.1])

        batch_idxs, values = test_map.get(state_id=0, action=1)
        self.assertEqual(batch_idxs, [2])
        self.assertEqual(values, [2.1])

        batch_idxs, values = test_map.get(state_id=1, action=0)
        self.assertEqual(batch_idxs, [1])
        self.assertEqual(values, [3.1])


class TestTrainingHistoryDatabase(unittest.TestCase):
    def setUp(self):
        test_utils.create_temp_dir()
        with object_logging.ObjectLogManager(
            dirname=test_utils.object_logging_dir()
        ) as object_log_manager:
            test_utils.get_trainer(max_steps=5).fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager,
                    debug_td_error=True,
                    debug=True,
                    max_episode_length=2,
                    initial_memories_count=9,
                )
            )

        self.training_history_database = object_log_readers.TrainingHistoryDatabase(
            dirname=test_utils.object_logging_dir()
        )

    def tearDown(self):
        test_utils.delete_temp_dir()

    def test_nA(self):
        self.assertEqual(self.training_history_database.nA, 3)

    @parameterized.expand(
        [
            (
                "singular",
                1,
            ),
            (
                "multiple",
                2,
            ),
            (
                "all implicit",
                None,
            ),
            (
                "all explicit",
                3,
            ),
            ("too many", 100),
        ]
    )
    def test_get_states_by_visit_count(self, name, n):
        max_possible_states = 3

        state_with_id_0 = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
        state_with_id_1 = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
        state_with_id_2 = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.0],
        ]

        expected_states = [state_with_id_0, state_with_id_2, state_with_id_1]
        expected_state_ids = [0, 1, 2]
        expected_visit_counts = [9, 3, 2]

        expected_n = min(max_possible_states, n) if n else max_possible_states

        visited_states = self.training_history_database.get_states_by_visit_count(n)

        self.assertEqual(len(visited_states), expected_n)
        for i, (expected_state_id, expected_state, expected_visit_count) in enumerate(
            zip(expected_state_ids[:n], expected_states[:n], expected_visit_counts[:n])
        ):
            self.assertEqual(visited_states[i].state_id, expected_state_id)
            self.assertEqual(visited_states[i].state.tolist(), expected_state)
            self.assertEqual(visited_states[i].visit_count, expected_visit_count)


    def test_get_td_errors(self):
        """Ensure that for all batches, certain states will log td errors for all actions to verify the shape of the response."""

        _, td_errors = self.training_history_database.get_td_errors(5, 5)
        self.assertEqual(len(td_errors), 0)

        state_ids = list(range(3))
        actions = [0, 1, 2]
        for (state_id, action) in itertools.product(state_ids, actions):
            batch_idxs, td_errors = self.training_history_database.get_td_errors(
                state_id=state_id, action=action
            )
            self.assertEqual(len(td_errors), 5)
            self.assertTrue(
                all([isinstance(td_error, float) for td_error in td_errors])
            )

    @parameterized.expand(
        [
            (None, None, 5),
            (0, 3, 4),
            (3, 4, 2),
        ]
    )
    def test_get_replay_buffer_state_counts(
        self,
        start_batch_idx: Optional[int],
        end_batch_idx: Optional[int],
        length_of_replay_buffer_state_counts,
    ):
        """Ensure that for different batch ranges, the replay buffer state counts will be logged."""
        (
            _,
            replay_buffer_state_counts,
        ) = self.training_history_database.get_replay_buffer_state_counts(
            start_batch_idx, end_batch_idx
        )
        self.assertEqual(
            len(replay_buffer_state_counts), length_of_replay_buffer_state_counts
        )
        previous_replay_buffer_total_states = 0
        for replay_buffer_batch_state_count in replay_buffer_state_counts:
            # Check that the previous replay buffer total states is incremented by one when the replay buffer is not at capacity.
            total_states = sum(
                count for _, count in replay_buffer_batch_state_count.items()
            )
            if previous_replay_buffer_total_states:
                self.assertEqual(previous_replay_buffer_total_states + 1, total_states)
            previous_replay_buffer_total_states = total_states

    def test_get_q_values_and_q_target_values(self):
        """Spot check a few state/action pairs to verify the shape of the response."""

        _, q_values = self.training_history_database.get_q_values(5, 5)
        self.assertEqual(len(q_values), 0)
        _, q_target_values = self.training_history_database.get_q_target_values(5, 5)
        self.assertEqual(len(q_target_values), 0)

        expected_batch_idx = list(range(5))
        for state_id, action in itertools.product(range(3), range(3)):
            batch_idxs_q, q_values = self.training_history_database.get_q_values(
                state_id, action
            )
            self.assertEqual(batch_idxs_q, expected_batch_idx)
            self.assertTrue(all([isinstance(q_value, float) for q_value in q_values]))

            (
                batch_idxs_q_target,
                q_target_values,
            ) = self.training_history_database.get_q_target_values(state_id, action)
            self.assertEqual(batch_idxs_q_target, expected_batch_idx)
            self.assertTrue(
                all(
                    [
                        isinstance(q_target_value, float)
                        for q_target_value in q_target_values
                    ]
                )
            )

            self.assertTrue(
                all(
                    [
                        q_value != q_target_value
                        for q_value, q_target_value in zip(q_values, q_target_values)
                    ]
                )
            )

    @parameterized.expand(
        [
            ("start filter", "get_td_errors", 2, None, 3),
            ("end filter", "get_td_errors", None, 3, 4),
            ("both filters", "get_td_errors", 2, 3, 2),
            ("start filter", "get_q_values", 2, None, 3),
            ("end filter", "get_q_values", None, 3, 4),
            ("both filters", "get_q_values", 2, 3, 2),
            ("start filter", "get_q_target_values", 2, None, 3),
            ("end filter", "get_q_target_values", None, 3, 4),
            ("both filters", "get_q_target_values", 2, 3, 2),
        ]
    )
    def test_getters_with_batch_idx_filters(
        self,
        name,
        getter_fn_name,
        start_batch_idx,
        end_batch_idx,
        expected_entry_count,
    ):
        """Verifies batch filter indices prune entries considered."""

        getter_fn = getattr(self.training_history_database, getter_fn_name)

        batch_idxs, values = getter_fn(
            state_id=0,
            action=0,
            start_batch_idx=start_batch_idx,
            end_batch_idx=end_batch_idx,
        )
        self.assertEqual(len(values), expected_entry_count)
        self.assertTrue(all([isinstance(value, float) for value in values]))


def _log_entries(entries: List[Any], buffer_size: int) -> None:
    object_logger = object_logging.ObjectLogger(
        dirname=test_utils.TMP_DIR,
        log_filename=_LOG_FILENAME_0,
        buffer_size=buffer_size,
    )
    for entry in entries:
        object_logger.log(entry)
    object_logger.close()


class TestReadObjectLog(unittest.TestCase):
    def setUp(self):
        test_utils.create_temp_dir()

    def tearDown(self):
        test_utils.delete_temp_dir()

    @parameterized.expand(
        [
            (
                "singular",
                1,
            ),
            (
                "multiple",
                2,
            ),
        ]
    )
    def test_read_object_log(self, name, buffer_size):
        """Verifies seamless iteration of the log independent of buffer_size"""

        expected_entries = ["a", "b", "c"]
        _log_entries(expected_entries, buffer_size)

        logged_entries = list(
            object_log_readers.read_object_log(
                os.path.join(test_utils.TMP_DIR, _LOG_FILENAME_0)
            )
        )
        self.assertEqual(expected_entries, logged_entries)


class TestActionInversionDatabase(unittest.TestCase):
    def setUp(self):
        test_utils.create_temp_dir()
        with object_logging.ObjectLogManager(
            dirname=test_utils.object_logging_dir()
        ) as object_log_manager:
            test_utils.get_trainer(max_steps=12).fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager,
                    env_width=4,
                    debug_action_inversion_checker=True,
                )
            )

        self._action_inversion_database = object_log_readers.ActionInversionDatabase(
            dirname=test_utils.object_logging_dir()
        )

    def tearDown(self):
        test_utils.delete_temp_dir()

    def test_get_divergences(self):
        # Iterates through various start and end combinations without
        # re-running the expensive setUp each time.
        expected = [
            object_log_readers.DivergenceEntry(
                batch_idx=6, convergence_run_length=5, divergence_magnitude=2
            ),
            object_log_readers.DivergenceEntry(
                batch_idx=10, convergence_run_length=2, divergence_magnitude=1
            ),
        ]

        test_cases = [
            ("all", None, None, expected),
            ("all start midrun", 5, None, expected),
            ("all start endpoint", 6, None, expected),
            ("all end endpoint", None, 10, expected),
            ("all end midrun", None, 11, expected),
            ("all start end", 6, 10, expected),
            ("filter start", 7, None, expected[1:]),
            ("filter end", None, 9, expected[:1]),
            ("filter start end", 7, 9, []),
        ]
        for (
            name,
            start_batch_idx,
            end_batch_idx,
            expected_divergences,
        ) in test_cases:
            divergences = self._action_inversion_database.get_divergences(
                start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
            )
            self.assertEqual(divergences, expected_divergences)

    def test_get_incidence_rate(self):
        # Iterates through various start and end combinations without
        # re-running the expensive setUp each time.
        test_cases = [
            ("all", None, None, [6, 7, 10, 11], [2, 2, 1, 1]),
            ("all start only", 6, None, [6, 7, 10, 11], [2, 2, 1, 1]),
            ("all end only", None, 11, [6, 7, 10, 11], [2, 2, 1, 1]),
            ("all start end", 6, 11, [6, 7, 10, 11], [2, 2, 1, 1]),
            ("filter start", 7, None, [7, 10, 11], [2, 1, 1]),
            ("filter end", None, 10, [6, 7, 10], [2, 2, 1]),
            ("filter start end", 7, 9, [7], [2]),
        ]
        for (
            name,
            start_batch_idx,
            end_batch_idx,
            expected_batch_idxs,
            expected_incidence_rate,
        ) in test_cases:
            (
                batch_idxs,
                incidence_rate,
            ) = self._action_inversion_database.get_incidence_rate(
                start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
            )
            self.assertEqual(batch_idxs, expected_batch_idxs)
            self.assertEqual(incidence_rate, expected_incidence_rate)

    def test_get_reports(self):
        batch_idx = 7
        reports = self._action_inversion_database.get_reports(batch_idx=batch_idx)
        # Ensure the same state isn't being returned for all reports.
        states = set()
        self.assertEqual(len(reports), 2)
        for report, state in reports:
            self.assertEqual(report.batch_idx, batch_idx)
            self.assertIsInstance(state, torch.Tensor)
            state_as_string = str(state)
            self.assertNotIn(state_as_string, states)
            states.add(state_as_string)


if __name__ == "__main__":
    unittest.main()
