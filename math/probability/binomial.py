#!/usr/bin/env python3
""" This module contains the binomial function"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate p as the proportion of successes in the data
            p = sum(data) / len(data)
            # n is the rounded total number of trials, which is the length of the data list
            n = round(len(data))
            # Recalculate p with the rounded n to ensure consistency
            p = sum(data) / n
            self.n = n
            self.p = float(p)
