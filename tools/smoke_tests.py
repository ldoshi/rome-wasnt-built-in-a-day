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
from tools import training_viewer

from bridger.logging import object_logging
from bridger.logging import object_log_readers
from bridger.logging import log_entry


_OBJECT_LOGGING_DIR_0 = "tmp_object_logging_dir_0"
_OBJECT_LOGGING_DIR_1 = "tmp_object_logging_dir_1"
_TRAINING_BATCH_LOG = "training_batch"
_STATE_NORMALIZED_LOG = "state_normalized"
_ACTION_INVERSION_REPORT_LOG = "action_inversion_report"


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

    def test_training_viewer(self):
        """Verifies training viewer can build."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_0
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(object_log_manager=object_log_manager, debug=True)
            )

        training_viewer.plot_training_data(log_dir=_OBJECT_LOGGING_DIR_0, num_states=5)

    def test_action_inversion_analyzer(self):
        """Verifies action inversion analyzer can build."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR_0
        ) as object_log_manager:
            test_utils.get_trainer().fit(
                test_utils.get_model(object_log_manager=object_log_manager, debug_action_inversion_checker=True)
            )

        analyzer = ActionInversionAnalyzer(
            action_inversion_log=os.path.join( _OBJECT_LOGGING_DIR_0, _ACTION_INVERSION_REPORT_LOG  ),
            state_normalized_log=os.path.join(  _OBJECT_LOGGING_DIR_0, _STATE_NORMALIZED_LOG )
        )

        analyzer.show_incidence_rate()
        analyzer.show_incidence_rate(start_batch_idx=3)
        analyzer.show_incidence_rate(end_batch_idx=6)

        analyzer.show_divergences()
        analyzer.show_divergences(start_batch_idx=3)
        analyzer.show_divergences(end_batch_idx=6)

        analyzer.print_divergences()
        analyzer.print_divergences(start_batch_idx=3)
        analyzer.print_divergences(end_batch_idx=6)
        analyzer.print_divergences(n=6)
        analyzer.print_divergences(sort_by_convergence_run_length=True)
        analyzer.print_divergences(sort_by_divergence_magnitude=True)
        analyzer.print_divergences(sort_by_convergence_run_length=True, sort_by_divergence_magnitude=True)

        
        

        



if __name__ == "__main__":
    unittest.main()
