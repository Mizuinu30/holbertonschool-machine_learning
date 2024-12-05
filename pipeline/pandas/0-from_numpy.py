#!/usr/bin/env python3
""" Defines a function that creates a pandas DataFrame from a numpy array """

import pandas as pd


def from_numpy(array):
    """ Creates a pandas DataFrame from a numpy array """
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
                "Y", "Z"]
    column_labels = []
    for i in range(len(array.shape[0])):
        column_labels.append(alphabet[i])
    df = pd.DataFrame(array, columns=column_labels)
    return df
