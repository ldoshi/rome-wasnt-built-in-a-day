"""Tests for object_log_cache."""
import unittest

import shutil

from bridger import test_utils
from bridger.logging import object_logging
from bridger.logging import object_log_readers
from bridger.logging import log_entry

from tools.web import object_log_cache

_OBJECT_LOGGING_DIR = "tmp_object_logging_dir"


class ObjectLogCacheTest(unittest.TestCase):
    """Verifies cache loading behavior."""

    def tearDown(self):
        # TODO: Make a more coherent plan for writing test output to a
        # temp dir and retaining it on failure.
        shutil.rmtree("lightning_logs", ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR, ignore_errors=True)

    def test_get_file_does_not_exist(self):
        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertIsNone(cache.get(object_log_cache.ACTION_INVERSION_DATABASE_KEY))
        self.assertEqual(
            cache.key_miss_counts[object_log_cache.ACTION_INVERSION_DATABASE_KEY],
            1,
        )
        self.assertEqual(
            cache.key_hit_counts[object_log_cache.ACTION_INVERSION_DATABASE_KEY],
            0,
        )

        self.assertIsNone(cache.get(object_log_cache.ACTION_INVERSION_DATABASE_KEY))
        self.assertEqual(
            cache.key_miss_counts[object_log_cache.ACTION_INVERSION_DATABASE_KEY],
            2,
        )
        self.assertEqual(
            cache.key_hit_counts[object_log_cache.ACTION_INVERSION_DATABASE_KEY],
            0,
        )

    def test_get_illegal_key(self):
        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertRaisesRegex(ValueError, "Unsupported", cache.get, key="illegal")

    def test_get_and_load(self):
        """Verifies cache loads files once."""

        max_steps = 1
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            test_utils.get_trainer(max_steps=max_steps).fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager,
                    debug=True,
                    env_width=4,
                    max_episode_length=10,
                    initial_memories_count=1,
                )
            )

        cache = object_log_cache.ObjectLogCache(log_dir=_OBJECT_LOGGING_DIR)

        self.assertIsNotNone(cache.get(object_log_cache.TRAINING_HISTORY_DATABASE_KEY))
        self.assertEqual(
            cache.key_miss_counts[object_log_cache.TRAINING_HISTORY_DATABASE_KEY], 1
        )
        self.assertEqual(
            cache.key_hit_counts[object_log_cache.TRAINING_HISTORY_DATABASE_KEY], 0
        )

        self.assertIsNotNone(cache.get(object_log_cache.TRAINING_HISTORY_DATABASE_KEY))
        self.assertEqual(
            cache.key_miss_counts[object_log_cache.TRAINING_HISTORY_DATABASE_KEY], 1
        )
        self.assertEqual(
            cache.key_hit_counts[object_log_cache.TRAINING_HISTORY_DATABASE_KEY], 1
        )

    def test_loaders_smoke_test(self):
        """Verifies each loader produces a reasonably shaped result."""
        max_steps = 4
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
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
            object_log_cache.ACTION_INVERSION_DATABASE_KEY
        )
        self.assertEqual(len(action_inversion_database.get_incidence_rate()), 2)

        training_history_database = cache.get(
            object_log_cache.TRAINING_HISTORY_DATABASE_KEY
        )
        self.assertEqual(len(training_history_database.get_states_by_visit_count()), 2)


if __name__ == "__main__":
    unittest.main()
