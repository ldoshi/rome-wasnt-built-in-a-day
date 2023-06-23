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
_SIBYL_TEMP_DIR = "tmp_sibyl_test"


def _generate_logs(
    experiment_name: str, max_steps: int, debug_action_inversion_checker: bool = False
) -> None:
    with object_logging.ObjectLogManager(
        dirname=object_log_cache.get_experiment_data_dir(
            log_dir=_OBJECT_LOGGING_DIR, experiment_name=experiment_name
        )
    ) as object_log_manager:
        test_utils.get_trainer(max_steps=max_steps).fit(
            test_utils.get_model(
                object_log_manager=object_log_manager,
                debug=True,
                debug_action_inversion_checker=debug_action_inversion_checker,
                env_width=4,
                max_episode_length=1,
                initial_memories_count=1,
            )
        )


class ObjectLogCacheTest(unittest.TestCase):
    """Verifies cache loading behavior."""

    def setUp(self):
        self._cache = object_log_cache.ObjectLogCache(
            log_dir=_OBJECT_LOGGING_DIR, temp_dir=_SIBYL_TEMP_DIR
        )

    def tearDown(self):
        # TODO: Make a more coherent plan for writing test output to a
        # temp dir and retaining it on failure.
        shutil.rmtree("lightning_logs", ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR, ignore_errors=True)
        shutil.rmtree(_SIBYL_TEMP_DIR, ignore_errors=True)

    def test_get_file_does_not_exist(self):

        self.assertRaisesRegex(
            FileNotFoundError,
            "action_inversion_report",
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )

        cache_key = object_log_cache._make_cache_key(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )
        self.assertEqual(
            self._cache.miss_counts[cache_key],
            1,
        )
        self.assertEqual(
            self._cache.hit_counts[cache_key],
            0,
        )

        self.assertRaisesRegex(
            FileNotFoundError,
            "action_inversion_report",
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )

        self.assertEqual(
            self._cache.miss_counts[cache_key],
            2,
        )
        self.assertEqual(
            self._cache.hit_counts[cache_key],
            0,
        )

    def test_get_illegal_key_and_experiment_name_data_is_different(self):
        self.assertRaises(
            FileNotFoundError,
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaises(
            FileNotFoundError,
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )

        max_steps = 1
        _generate_logs(experiment_name=_EXPERIMENT_NAME_0, max_steps=max_steps)

        # By not raising, we know _EXPERIMENT_NAME_0 now
        # exists. _EXPERIMENT_NAME_1 still does not exist and does
        # raise.
        self._cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaises(
            FileNotFoundError,
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertRaisesRegex(
            ValueError,
            "Unsupported",
            self._cache.get,
            experiment_name=_EXPERIMENT_NAME_0,
            data_key="illegal",
        )

        _generate_logs(experiment_name=_EXPERIMENT_NAME_1, max_steps=max_steps)

        # By not raising, we know _EXPERIMENT_NAME_1 now exists.
        self._cache.get(
            experiment_name=_EXPERIMENT_NAME_1,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )

    def test_get_and_load(self):
        """Verifies cache loads files once."""

        _generate_logs(experiment_name=_EXPERIMENT_NAME_0, max_steps=1)

        self.assertIsNotNone(
            self._cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        cache_key = object_log_cache._make_cache_key(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertEqual(self._cache.miss_counts[cache_key], 1)
        self.assertEqual(self._cache.hit_counts[cache_key], 0)
        self.assertEqual(self._cache.load_database_miss_counts[cache_key], 1)
        self.assertEqual(self._cache.load_database_hit_counts[cache_key], 0)

        self.assertIsNotNone(
            self._cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        self.assertEqual(self._cache.miss_counts[cache_key], 1)
        self.assertEqual(self._cache.hit_counts[cache_key], 1)
        self.assertEqual(self._cache.load_database_miss_counts[cache_key], 1)
        self.assertEqual(self._cache.load_database_hit_counts[cache_key], 0)

    def test_load(self):
        """Verifies data is loaded from log only once across new cache instances."""

        _generate_logs(experiment_name=_EXPERIMENT_NAME_0, max_steps=1)

        cache_key = object_log_cache._make_cache_key(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )

        self.assertIsNotNone(
            self._cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        self.assertEqual(self._cache.load_database_miss_counts[cache_key], 1)
        self.assertEqual(self._cache.load_database_hit_counts[cache_key], 0)

        cache = object_log_cache.ObjectLogCache(
            log_dir=_OBJECT_LOGGING_DIR, temp_dir=_SIBYL_TEMP_DIR
        )
        self.assertIsNotNone(
            cache.get(
                experiment_name=_EXPERIMENT_NAME_0,
                data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
            )
        )
        self.assertEqual(cache.load_database_miss_counts[cache_key], 0)
        self.assertEqual(cache.load_database_hit_counts[cache_key], 1)

    def test_loaders_smoke_test(self):
        """Verifies each loader produces a reasonably shaped result."""
        max_steps = 4
        _generate_logs(
            experiment_name=_EXPERIMENT_NAME_0,
            max_steps=max_steps,
            debug_action_inversion_checker=True,
        )

        action_inversion_database = self._cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
        )
        self.assertEqual(len(action_inversion_database.get_incidence_rate()), 2)

        training_history_database = self._cache.get(
            experiment_name=_EXPERIMENT_NAME_0,
            data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
        )
        self.assertEqual(len(training_history_database.get_states_by_visit_count()), 1)

    def test_convert_logs_to_saved_databases(self):
        """Verifies logs are converted before load."""
        max_steps = 4
        _generate_logs(
            experiment_name=_EXPERIMENT_NAME_0,
            max_steps=max_steps,
            debug_action_inversion_checker=True,
        )

        cache_keys = [
            object_log_cache._make_cache_key(_EXPERIMENT_NAME_0, data_key)
            for data_key in object_log_cache._LOADERS.keys()
        ]
        for cache_key in cache_keys:
            self.assertEqual(self._cache.load_database_miss_counts[cache_key], 0)
            self.assertEqual(self._cache.load_database_hit_counts[cache_key], 0)

        self._cache.convert_logs_to_saved_databases([_EXPERIMENT_NAME_0])

        for cache_key in cache_keys:
            self._cache.get(experiment_name=cache_key[0], data_key=cache_key[1])

        for cache_key in cache_keys:
            self.assertEqual(self._cache.load_database_miss_counts[cache_key], 0)
            self.assertEqual(self._cache.load_database_hit_counts[cache_key], 1)


if __name__ == "__main__":
    unittest.main()
