"""Tests for plot_utils."""
import pandas as pd
import unittest

from parameterized import parameterized

from tools.web import plot_utils


class TestPlotUtils(unittest.TestCase):
    """Testing plot utility functions."""

    @parameterized.expand(
        [
            ("one", 1, [0]),
            ("endpoints", 2, [0, 4]),
            ("spaced out", 3, [0, 2, 4]),
            ("all", 5, [0, 1, 2, 3, 4]),
            ("too many", 6, [0, 1, 2, 3, 4]),
        ]
    )
    def test_downsample(self, name, n, expected_indices):
        df = pd.DataFrame(
            [
                ("e", 10),
                ("d", 20),
                ("c", 30),
                ("b", 40),
                ("a", 50),
            ]
        )
        sampled_df = plot_utils.downsample(df, n)
        self.assertTrue(sampled_df.equals(df.iloc[expected_indices]))


if __name__ == "__main__":
    unittest.main()
