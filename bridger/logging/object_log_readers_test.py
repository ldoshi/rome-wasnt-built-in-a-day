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
from bridger.logging import object_log_readers

_TMP_DIR = "tmp/nested_tmp"


def create_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)


def delete_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    shutil.rmtree(path.parts[0], ignore_errors=True)


_LOG_FILENAME_0 = "log_filename_0"
_LOG_FILENAME_1 = "log_filename_1"



def _log_entries(entries: List[Any], buffer_size: int) -> None:
    object_logger = object_logging.ObjectLogger(
        dirname=_TMP_DIR, log_filename=_LOG_FILENAME_0, buffer_size=buffer_size
    )
    for entry in entries:
        object_logger.log(entry)
    object_logger.close()


class TestReadObjectLog(unittest.TestCase):
    def setUp(self):
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()

    @parameterized.expand([(1,), (2,)])
    def test_read_object_log(self, buffer_size):
        """Verifies seamless iteration of the log independent of buffer_size"""

        expected_entries = ["a", "b", "c"]
        _log_entries(expected_entries, buffer_size)

        logged_entries = list(object_log_readers.read_object_log(_TMP_DIR, _LOG_FILENAME_0))
        self.assertEqual(expected_entries, logged_entries)


if __name__ == "__main__":
    unittest.main()
