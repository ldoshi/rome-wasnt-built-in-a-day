"""Tests to ensure tools are not broken.

These smoke tests ensure the basic usage flow for tools does not
produce any build or run failures. These are not rigorous tests of the
tools' specific functionality.

Most tools are light weight and lightly tested for functionality, if
at all.

"""

import os
import shutil
import unittest

from bridger import test_utils

from tools import training_batch_comparison_tool

from bridger.logging import object_logging
from bridger.logging import object_log_readers
from bridger.logging import log_entry


_OBJECT_LOGGING_DIR_0 = "tmp_object_logging_dir_0"
_OBJECT_LOGGING_DIR_1 = "tmp_object_logging_dir_1"
_TRAINING_BATCH_LOG = "training_batch"


class ToolsSmokeTest(unittest.TestCase):
    """Verifies tool usage flow."""

    def tearDown(self):
        # TODO: Make a more coherent plan for writing test output to a temp dir
        #       and retaining it on failure
        shutil.rmtree("lightning_logs", ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR_0, ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR_1, ignore_errors=True)

    def test_training_batch_comparison_tool_matching(self):
        """Verifies comparison two training batches that match."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_0
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(object_log_manager=object_log_manager, debug=True)
            )

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_1
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(object_log_manager=object_log_manager, debug=True)
            )

        self.assertTrue(
            training_batch_comparison_tool.compare_training_batches(
                expected_log=os.path.join(_OBJECT_LOGGING_DIR_0, _TRAINING_BATCH_LOG),
                test_log=os.path.join(_OBJECT_LOGGING_DIR_1, _TRAINING_BATCH_LOG),
            )
        )

    def test_training_batch_comparison_tool_different(self):
        """Verifies comparison two training batches that differ."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_0
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(object_log_manager=object_log_manager, debug=True)
            )

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_1
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(
                    object_log_manager=object_log_manager, batch_size=4, debug=True
                )
            )

        self.assertFalse(
            training_batch_comparison_tool.compare_training_batches(
                expected_log=os.path.join(_OBJECT_LOGGING_DIR_0, _TRAINING_BATCH_LOG),
                test_log=os.path.join(_OBJECT_LOGGING_DIR_1, _TRAINING_BATCH_LOG),
            )
        )


if __name__ == "__main__":
    unittest.main()
