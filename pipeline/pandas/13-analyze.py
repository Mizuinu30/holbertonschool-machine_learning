#!/usr/bin/env python3
"""
This module analyzes data in a DataFrame
"""


def analyze(df):
    """ This function analyzes data in a DataFrame """

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    return df.describe()
