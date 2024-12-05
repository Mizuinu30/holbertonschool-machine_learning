#!/usr/bin/env python3
"""This module fills missing values in a dataframe"""


def fill(df):
    """ This function fills missing values in a dataframe """
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    if "Close" in df.columns:
        df["Close"] = df["Close"].ffill()

    for column in ["high", "Low", "Open"]:
        if column in df.columns:
            df[column]  = df[column].fillna(df["Close"])

    for column in ["Volume_(BTC)", "Volume_(Currency)"]:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df
