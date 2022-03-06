import dataclasses
import pickle
import math
import unittest
import os
import pathlib
import shutil
import torch
from typing import Any, List

from parameterized import parameterized

from bridger.logging import log_entry
from bridger.logging import object_logging

_TMP_DIR = "tmp/nested_tmp"


def create_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)


def delete_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    shutil.rmtree(path.parts[0], ignore_errors=True)


_LOG_FILENAME_0 = "log_filename_0"
_LOG_FILENAME_1 = "log_filename_1"


class TestObjectLogManager(unittest.TestCase):
    def tearDown(self):
        # The _TMP_DIR is created by the ObjectLogManager init.
        delete_temp_dir()

    def test_object_log_manager_basic(self):
        expected_log_entries_0 = ["a", "b", "c"]
        expected_log_entries_1 = ["d"]

        with object_logging.ObjectLogManager(dirname=_TMP_DIR) as logger:
            for log_entry in expected_log_entries_0:
                logger.log(_LOG_FILENAME_0, log_entry)

            for log_entry in expected_log_entries_1:
                logger.log(_LOG_FILENAME_1, log_entry)

        logged_entries_0 = list(
            object_logging.read_object_log(_TMP_DIR, _LOG_FILENAME_0)
        )
        self.assertEqual(expected_log_entries_0, logged_entries_0)

        logged_entries_1 = list(
            object_logging.read_object_log(_TMP_DIR, _LOG_FILENAME_1)
        )
        self.assertEqual(expected_log_entries_1, logged_entries_1)


class TestLoggerAndNormalizer(unittest.TestCase):
    def setUp(self):
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()

    @parameterized.expand(
        [
            ("Hashable Log Entry", int, None, 1, 5),
            ("Non-Hashable Log Entry", list, str, [1], [2, 5]),
        ]
    )
    def test_get_logged_object_id(
        self, name, log_entry_object_class, make_hashable_fn, object_0, object_1
    ):
        with object_logging.ObjectLogManager(dirname=_TMP_DIR) as logger:
            normalizer = object_logging.LoggerAndNormalizer(
                log_filename=_LOG_FILENAME_0,
                object_log_manager=logger,
                log_entry_object_class=log_entry_object_class,
                make_hashable_fn=make_hashable_fn,
            )
            self.assertEqual(normalizer.get_logged_object_id(object_0), 0)
            self.assertEqual(normalizer.get_logged_object_id(object_0), 0)
            self.assertEqual(normalizer.get_logged_object_id(object_1), 1)
            self.assertEqual(normalizer.get_logged_object_id(object_0), 0)

        expected_entries = [
            log_entry.NormalizedLogEntry(id=0, object=object_0),
            log_entry.NormalizedLogEntry(id=1, object=object_1),
        ]
        logged_entries = list(object_logging.read_object_log(_TMP_DIR, _LOG_FILENAME_0))
        self.assertEqual(logged_entries, expected_entries)

    def test_get_logged_object_by_id(      self ):
        object = 1
        with object_logging.ObjectLogManager(dirname=_TMP_DIR) as logger:
            normalizer = object_logging.LoggerAndNormalizer(
                log_filename=_LOG_FILENAME_0,
                object_log_manager=logger,
                log_entry_object_class=int
            )
            self.assertEqual(normalizer.get_logged_object_id(object), 0)
            self.assertEqual(normalizer.get_logged_object_by_id(object_id=0), object)
            self.assertRaises(ValueError, normalizer.get_logged_object_by_id, object_id=1)


    def test_logging_incorrect_type(self):
        with object_logging.ObjectLogManager(dirname=_TMP_DIR) as logger:
            normalizer = object_logging.LoggerAndNormalizer(
                log_filename=_LOG_FILENAME_0,
                object_log_manager=logger,
                log_entry_object_class=dict,
            )
            self.assertRaises(ValueError, normalizer.get_logged_object_id, object=[])

    def test_illegal_init_configuration(self):
        with object_logging.ObjectLogManager(dirname=_TMP_DIR) as logger:
            self.assertRaises(ValueError, object_logging.LoggerAndNormalizer, log_filename=_LOG_FILENAME_0,
                object_log_manager=logger,
                log_entry_object_class=torch.Tensor)


def _log_entries(entries: List[Any], buffer_size: int) -> None:
    object_logger = object_logging.ObjectLogger(
        dirname=_TMP_DIR, log_filename=_LOG_FILENAME_0, buffer_size=buffer_size
    )
    for entry in entries:
        object_logger.log(entry)
    object_logger.close()


class TestObjectLogger(unittest.TestCase):
    def setUp(self):
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()

    @parameterized.expand([(1,), (2,)])
    def test_object_logger(self, buffer_size):
        """Verifies object logging with varying buffer sizes.

        The buffer_size of 1 tests the base case. The buffer_size of 2
        tests the case where a) the buffer needs to be flushed on exit
        and b) the final pickled entry is has fewer elements than the
        previous entries, which contain buffer_size elements each.
        """

        test_filepath = os.path.join(_TMP_DIR, _LOG_FILENAME_0)
        entries = ["a", "b", "c"]
        _log_entries(entries, buffer_size)

        with open(test_filepath, "rb") as f:
            for i in range(math.ceil(len(entries) / buffer_size)):
                buffer = pickle.load(f)
                self.assertEqual(
                    buffer, entries[i * buffer_size : (i + 1) * buffer_size]
                )

            self.assertRaises(EOFError, pickle.load, f)

    @parameterized.expand([(1,), (2,)])
    def test_read_object_log(self, buffer_size):
        """Verifies seamless iteration of the log independent of buffer_size"""

        expected_entries = ["a", "b", "c"]
        _log_entries(expected_entries, buffer_size)

        logged_entries = list(object_logging.read_object_log(_TMP_DIR, _LOG_FILENAME_0))
        self.assertEqual(expected_entries, logged_entries)


if __name__ == "__main__":
    unittest.main()
