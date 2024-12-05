#!/usr/bin/env python3
"""This modlue prunes a dataframe to only contain the columns"""


def prune(df):
    """This function prunes a dataframe to only contain the columns"""

    # Prune the DF to only contain the columns 'Close'

    return df.dropna(subset=['Close'])
