#!/usr/bin/env python3
"""This module sorts a dataframe in descending order"""


def high(df):
    """ A function that sorts a dataframe in descending order """

    return df.sort_values(by='High', ascending=False)
