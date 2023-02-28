"""Tests for experiment_runner."""
import unittest

from typing import Any, List

import itertools
import json
import os
from parameterized import parameterized

from tools import experiment_runner

_TEST_DATA_DIR = "test_data"
_BASIC_CONFIG = "basic_config.json"
_BASIC_CONFIG_REORDERED = "basic_config_reordered.json"
_NAMELESS_CONFIG = "nameless_config.json"
_TWO_SWEEPS_CONFIG = "two_sweeps_config.json"
_TWO_SWEEPS_NAMELESS_REORDERED_CONFIG = "two_sweeps_nameless_reordered_config.json"

_EPSILON_DECAY_RULE = "--epsilon-decay-rule"
_INITIAL_MEMORIES_COUNT_KEY = "--initial-memories-count"
_EXPERIMENT_NAME_KEY = "--experiment-name"


def _load_config(filename: str) -> Any:
    with open(os.path.join(_TEST_DATA_DIR, filename)) as f:
        return json.load(f)


def _get_value(args: List[str], key: str) -> str:
    for i in range(len(args)):
        if args[i] == key:
            return args[i + 1]

    assert False, f"Unrecognized key: {key}"


class ExperimentRunnerTest(unittest.TestCase):
    """Verifies parameter sweeps from configs."""

    def setUp(self):
        self._args = []

        def execute_fn(args: List[str]) -> None:
            self._args.append(args)

        self._execute_fn = execute_fn

    @parameterized.expand(
        [
            ("basic config", _BASIC_CONFIG),
            ("basic config different flag order", _BASIC_CONFIG_REORDERED),
        ]
    )
    def test_execution_args_basic(self, name: str, config_filename: str):
        """Verifies execution args are built correctly."""

        config = _load_config(config_filename)
        experiment_runner.run_experiments(config=config, execute_fn=self._execute_fn)
        self.assertEqual(len(self._args), 1)
        self.assertEqual(
            self._args[0],
            [
                "--debug",
                "--epsilon-decay-rule",
                "geometric",
                "--initial-memories-count",
                "101",
                "--val-check-interval",
                "12",
                "--experiment-name",
                "basic_config",
            ],
        )

    def test_execution_args_nameless(self):
        config = _load_config(_NAMELESS_CONFIG)
        experiment_runner.run_experiments(config=config, execute_fn=self._execute_fn)
        self.assertEqual(len(self._args), 1)
        self.assertEqual(
            self._args[0], ["--env-width", "3", "--val-check-interval", "13"]
        )

    @parameterized.expand(
        [
            ("two sweeps config", _TWO_SWEEPS_CONFIG, "two_sweeps"),
            ("two sweeps nameless config", _TWO_SWEEPS_NAMELESS_REORDERED_CONFIG, ""),
        ]
    )
    def test_two_sweeps(self, name: str, config: str, experiment_name_prefix: str):
        config = _load_config(config)
        experiment_runner.run_experiments(config=config, execute_fn=self._execute_fn)
        self.assertEqual(len(self._args), 12)
        expected_value_tuples = itertools.product(
            ["--debug", "--no-debug"],
            ["geometric", "arithmetic"],
            ["0", "100", "10000"],
        )

        for args, expected_value_tuple in zip(self._args, expected_value_tuples):
            self.assertEqual(args[2], expected_value_tuple[0])
            self.assertEqual(
                _get_value(args, _EPSILON_DECAY_RULE), expected_value_tuple[1]
            )
            self.assertEqual(
                _get_value(args, _INITIAL_MEMORIES_COUNT_KEY), expected_value_tuple[2]
            )

            values = list(expected_value_tuple)
            values[0] = True if values[0] == "--debug" else False
            self.assertEqual(
                _get_value(args, _EXPERIMENT_NAME_KEY),
                experiment_runner._get_experiment_name(
                    experiment_name_prefix=experiment_name_prefix, values=values
                ),
            )


if __name__ == "__main__":
    unittest.main()
