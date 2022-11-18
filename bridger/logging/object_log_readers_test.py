import dataclasses
import pickle
import math
import itertools
import os
import unittest

import torch
from typing import Any, List

from parameterized import parameterized

from bridger.logging import log_entry
from bridger.logging import object_logging
from bridger.logging import object_log_readers
from bridger import test_utils


_LOG_FILENAME_0 = "log_filename_0_pytest"


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

    def test_actions_n(self):
        self.assertEqual(self.training_history_database.actions_n, 3)

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
            self.assertEqual(visited_states["state_id"].iloc[i], expected_state_id)
            self.assertEqual(visited_states["state"].iloc[i].tolist(), expected_state)
            self.assertEqual(
                visited_states["visit_count"].iloc[i], expected_visit_count
            )

    @parameterized.expand(
        [
            ("start filter", 2, None, [0, 1]),
            ("end filter", None, 3, [0, 1, 2]),
            ("both filters", 2, 3, [0]),
        ]
    )
    def test_get_states_by_visit_count_with_batch_index_filters(
        self, name, start_batch_index, end_batch_index, expected_state_ids
    ):
        """Verifies batch filter indices prune states considered."""
        visited_states = self.training_history_database.get_states_by_visit_count(
            start_batch_index=start_batch_index, end_batch_index=end_batch_index
        )

        self.assertEqual(list(visited_states["state_id"]), expected_state_ids)

    def test_get_td_errors(self):
        """Ensure that for all batches, certain states will log td errors for all actions to verify the shape of the response."""

        self.assertEqual(len(self.training_history_database.get_td_errors(5, 5)), 0)
        state_ids = list(range(3))
        actions = [0, 1, 2]
        for (state_id, action) in itertools.product(state_ids, actions):
            td_errors = self.training_history_database.get_td_errors(
                state_id=state_id, action=action
            )
            self.assertEqual(len(td_errors), 5)
            self.assertTrue(
                all([isinstance(td_error, float) for td_error in td_errors["td_error"]])
            )

    def test_get_q_values_and_q_target_values(self):
        """Spot check a few state/action pairs to verify the shape of the response."""

        self.assertEqual(len(self.training_history_database.get_q_values(5, 5)), 0)
        self.assertEqual(
            len(self.training_history_database.get_q_target_values(5, 5)), 0
        )

        expected_batch_idx = list(range(5))
        for state_id, action in itertools.product(range(3), range(3)):
            q_values = self.training_history_database.get_q_values(state_id, action)
            self.assertEqual(list(q_values["batch_idx"]), expected_batch_idx)
            self.assertTrue(
                all([isinstance(q_value, float) for q_value in q_values["q_value"]])
            )

            q_target_values = self.training_history_database.get_q_target_values(
                state_id, action
            )
            self.assertEqual(list(q_target_values["batch_idx"]), expected_batch_idx)
            self.assertTrue(
                all(
                    [
                        isinstance(q_target_value, float)
                        for q_target_value in q_target_values["q_target_value"]
                    ]
                )
            )

            self.assertNotEqual(
                list(q_values["q_value"]), list(q_target_values["q_target_value"])
            )

    @parameterized.expand(
        [
            ("start filter", "get_td_errors", "td_error", 2, None, 3),
            ("end filter", "get_td_errors", "td_error", None, 3, 4),
            ("both filters", "get_td_errors", "td_error", 2, 3, 2),
            ("start filter", "get_q_values", "q_value", 2, None, 3),
            ("end filter", "get_q_values", "q_value", None, 3, 4),
            ("both filters", "get_q_values", "q_value", 2, 3, 2),
            ("start filter", "get_q_target_values", "q_target_value", 2, None, 3),
            ("end filter", "get_q_target_values", "q_target_value", None, 3, 4),
            ("both filters", "get_q_target_values", "q_target_value", 2, 3, 2),
        ]
    )
    def test_getters_with_batch_index_filters(
        self,
        name,
        getter_fn_name,
        value_key,
        start_batch_index,
        end_batch_index,
        expected_entry_count,
    ):
        """Verifies batch filter indices prune entries considered."""

        getter_fn = getattr(self.training_history_database, getter_fn_name)

        values = getter_fn(
            state_id=0,
            action=0,
            start_batch_index=start_batch_index,
            end_batch_index=end_batch_index,
        )
        self.assertEqual(len(values), expected_entry_count)
        self.assertTrue(all([isinstance(value, float) for value in values[value_key]]))


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
        for name, start_batch_index, end_batch_index, expected_divergences in test_cases:
            divergences = self._action_inversion_database.get_divergences(
                start_batch_index=start_batch_index, end_batch_index=end_batch_index
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
            start_batch_index,
            end_batch_index,
            expected_batch_idxs,
            expected_incidence_rate,
        ) in test_cases:
            (
                batch_idxs,
                incidence_rate,
            ) = self._action_inversion_database.get_incidence_rate(
                start_batch_index=start_batch_index, end_batch_index=end_batch_index
            )
            self.assertEqual(batch_idxs, expected_batch_idxs)
            self.assertEqual(incidence_rate, expected_incidence_rate)


if __name__ == "__main__":
    unittest.main()
