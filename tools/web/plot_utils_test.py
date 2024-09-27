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

    def test_downsample_empty_df(self):
        df = pd.DataFrame([])
        sampled_df = plot_utils.downsample(df, n=1)
        self.assertTrue(sampled_df.empty)

    @parameterized.expand(
        [
            ("one", 1, [10]),
            ("endpoints", 2, [10, 50]),
            ("spaced out", 3, [10, 30, 50]),
            ("all", 5, [10, 20, 30, 40, 50]),
            ("too many", 6, [10, 20, 30, 40, 50]),
        ]
    )
    def test_downsample_list(self, name, n, expected):
        data = [10, 20, 30, 40, 50]
        sampled = plot_utils.downsample_list(data, n)
        self.assertEqual(sampled, expected)

    def test_downsample_empty_list(self):
        data = []
        sampled = plot_utils.downsample_list(data, n=1)
        self.assertFalse(sampled)


if __name__ == "__main__":
    unittest.main()
