"""A set of helpers for plotting data."""

from typing import Any, List

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
    if df.empty:
        return df
    return df.take(np.unique(np.round(np.linspace(0, len(df) - 1, n)).astype(int)))


def downsample_list(data: List[Any], n: int) -> List[Any]:
    """Downsamples a list to have at most n elements.

    The samples are taken equally spaced between the first and last
    elements. For n > 1, the first and last elements are always
    included in the sample. If the list has fewer than n elements, the
    list is returned as is.

    Args:
      data: The list from which to downsample.
      n: The target size for the downsampled list.

    Returns:
      A list with at most n elements.

    """
    if not data:
        return data
    return [
        data[i]
        for i in np.unique(np.round(np.linspace(0, len(data) - 1, n)).astype(int))
    ]
