"""Tests for object_log_cache."""
import unittest

import shutil

from bridger import test_utils
from bridger.logging_utils import object_logging
from bridger.logging_utils import object_log_readers
from bridger.logging_utils import log_entry

from tools.web import object_log_cache

_OBJECT_LOGGING_DIR = "tmp_object_logging_dir"
_EXPERIMENT_NAME_0 = "experiment_name_0"
_EXPERIMENT_NAME_1 = "experiment_name_1"


def _generate_logs(experiment_name: str, max_steps: int) -> None:
    with object_logging.ObjectLogManager(
        dirname=object_log_cache.get_experiment_data_dir(
            log_dir=_OBJECT_LOGGING_DIR, experiment_name=experiment_name
        )
    ) as object_log_manager:
        test_utils.get_trainer(max_steps=max_steps).fit(
            test_utils.get_model(
                object_log_manager=object_log_manager,
                debug=True,
                env_width=4,
                max_episode_length=1,
                initial_memories_count=1,
            )
        )


class ObjectLogCacheTest(unittest.TestCase):
    """Verifies cache loading behavior."""

    def tearDown(self):
        # TODO: Make a more coherent plan for writing test output to a
        # temp dir and retaining it on failure.
        shutil.rmtree("lightning_logs", ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR, ignore_errors=True)

    def test_get_file_does_not_exist(self):
        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertRaisesRegex(
            FileNotFoundError,
            "action_inversion_report",
            cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )

        cache_key = (_EXPERIMENT_NAME_0, object_log_cache.ACTION_INVERSION_DATABASE_KEY)
        self.assertEqual(
            cache.miss_counts[cache_key],
            1,
        )
        self.assertEqual(
            cache.hit_counts[cache_key],
            0,
        )

        self.assertRaisesRegex(
            FileNotFoundError,
            "action_inversion_report",
            cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )

        self.assertEqual(
            cache.miss_counts[cache_key],
            2,
        )
        self.assertEqual(
            cache.hit_counts[cache_key],
            0,
        )

    def test_get_illegal_key_and_experiment_name_data_is_different(self):
        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertRaises(
            FileNotFoundError,
            cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaises(
            FileNotFoundError,
            cache.get,
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )

        max_steps = 1
        _generate_logs(experiment_name=_EXPERIMENT_NAME_0, max_steps=max_steps)

        # By not raising, we know _EXPERIMENT_NAME_0 now
        # exists. _EXPERIMENT_NAME_1 still does not exist and does
        # raise.
        cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaises(
            FileNotFoundError,
            cache.get,
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaisesRegex(
            ValueError,
            "Unsupported",
            cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key="illegal",
        )

        _generate_logs(experiment_name=_EXPERIMENT_NAME_1, max_steps=max_steps)

        # By not raising, we know _EXPERIMENT_NAME_1 now exists.
        cache.get(
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )

    def test_get_and_load(self):
        """Verifies cache loads files once."""

        _generate_logs(experiment_name=_EXPERIMENT_NAME_0, max_steps=1)

        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertIsNotNone(
            cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        cache_key = (_EXPERIMENT_NAME_0, object_log_cache.TRAINING_HISTORY_DATABASE_KEY)
        self.assertEqual(cache.miss_counts[cache_key], 1)
        self.assertEqual(cache.hit_counts[cache_key], 0)

        self.assertIsNotNone(
            cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        self.assertEqual(cache.miss_counts[cache_key], 1)
        self.assertEqual(cache.hit_counts[cache_key], 1)

    def test_loaders_smoke_test(self):
        """Verifies each loader produces a reasonably shaped result."""
        max_steps = 4
        with object_logging.ObjectLogManager(
            dirname=object_log_cache.get_experiment_data_dir(
                log_dir=_OBJECT_LOGGING_DIR, experiment_name=_EXPERIMENT_NAME_0
            )
        ) as object_log_manager:
            test_utils.get_trainer(max_steps=max_steps).fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager,
                    debug_action_inversion_checker=True,
                    debug=True,
                    env_width=4,
                    max_episode_length=10,
                    initial_memories_count=1,
                )
            )

        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        action_inversion_database = cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )
        self.assertEqual(len(action_inversion_database.get_incidence_rate()), 2)

        training_history_database = cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertEqual(len(training_history_database.get_states_by_visit_count()), 2)

    def test_warm_simple(self):
        """Verifies cache loads files once."""

        max_steps = 1
        with object_logging.ObjectLogManager(
            dirname=object_log_cache.get_experiment_data_dir(
                log_dir=_OBJECT_LOGGING_DIR, experiment_name=_EXPERIMENT_NAME_0
            )
        ) as object_log_manager:
            test_utils.get_trainer(max_steps=max_steps).fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager,
                    debug=True,
                    env_width=4,
                    max_episode_length=1,
                    initial_memories_count=1,
                )
            )

        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertIsNotNone(
            cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        cache_key = (_EXPERIMENT_NAME_0, object_log_cache.TRAINING_HISTORY_DATABASE_KEY)
        self.assertEqual(cache.miss_counts[cache_key], 1)
        self.assertEqual(cache.hit_counts[cache_key], 0)

        self.assertIsNotNone(
            cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        self.assertEqual(cache.miss_counts[cache_key], 1)
        self.assertEqual(cache.hit_counts[cache_key], 1)


if __name__ == "__main__":
    unittest.main()
