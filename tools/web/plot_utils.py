"""A set of helpers for plotting data."""

import numpy as np
import pandas as pd


def downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Downsamples df to have at most n elements.

    The samples are taken equally spaced between df.iloc[0] and
    df.iloc[len(df)]. If df has fewer than n elements, df is returned
    as is.

    Args:
      df: The dataframe from which to downsample.
      n: The target size for the downsampled dataframe.

    Returns:
      A dataframe with at most n elements.
    """
    return df.take(np.unique(np.round(np.linspace(0, len(df) - 1, n)).astype(int)))
