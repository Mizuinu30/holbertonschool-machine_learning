#!/usr/bin/env python3
"""
This module contains the `from_numpy` function, which creates a pandas DataFrame
from a numpy ndarray with columns labeled alphabetically in uppercase.
"""

import pandas as pd

def from_numpy(array):
    """
    Creates a pandas DataFrame from a numpy ndarray.

    Parameters:
    array (np.ndarray): The input numpy array.

    Returns:
    pd.DataFrame: A DataFrame created from the input array, with columns labeled
                  alphabetically in uppercase (A, B, C, ...).

    Raises:
    ValueError: If the input is not a numpy ndarray or if the array has more
                than 26 columns.
    """
    if not isinstance(array, pd.np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")

    num_columns = array.shape[1]
    if num_columns > 26:
        raise ValueError("The array cannot have more than 26 columns.")

    column_labels = [chr(65 + i) for i in range(num_columns)]
    return pd.DataFrame(array, columns=column_labels)
