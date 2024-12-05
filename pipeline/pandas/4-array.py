#!/usr/bin/env python3
"""
New code updates the script to take the last 10 columns of High and Close
   and converts them into numpy.ndarray
"""

import pandas as pd


def array(df):
    """ This function takes the last 10 columns of High and Close and"""

    return df[["High", "Close"]].tail(10).to_numpy()
