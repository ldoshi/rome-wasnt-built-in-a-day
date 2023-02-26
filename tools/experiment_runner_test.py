"""Tests for experiment_runner."""
import unittest

from typing import Any, List

import json
import os
from parameterized import parameterized

from tools import experiment_runner

_TEST_DATA_DIR = "test_data"
_BASIC_CONFIG = "basic_config.json"
_BASIC_CONFIG_REORDERED = "basic_config_reordered.json"

def _load_config(filename: str) -> Any:
    with open(os.path.join(_TEST_DATA_DIR, filename)) as f:
        return json.load(f)

class ExperimentRunnerTest(unittest.TestCase):
    """Verifies parameter sweeps from configs."""

    def setUp(self):
        def execute_fn(args: List[str]) -> None:
            self._args = args

        self._execute_fn = execute_fn

    @parameterized.expand(
         [
             ("basic config", _BASIC_CONFIG),
             ("basic config different flag order", _BASIC_CONFIG_REORDERED)
         ]
    )
    def test_execution_args(self, name: str, config_filename: str):
        """Verifies execution args are built correctly."""
        
        config = _load_config(config_filename)
        experiment_runner.run_experiments(config=config, execute_fn=self._execute_fn)
        self.assertEqual(self._args, ['--experiment-name', 'basic_config', '--initial-memories-count', 101, '--val-check-interval', 12])
        

    # @parameterized.expand(
    #     [
    #         ("Single config", 1)
    #         ("Multiple config", 2)
    #     ]
    # )
    # def test_early_stopping(
    #     self,
    #     name: str,
    #     early_stopping_callback: list[Callback],
    # ):
    # Multi COUNT. and ordering for 2x sweeps.


if __name__ == "__main__":
    unittest.main()
