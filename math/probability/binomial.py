#!/usr/bin/env python3
"""This modluels contains the binomial distribution class"""


class Binomial:
    """This class represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial class constructor
        Args:
            data (List): List of the data to be used to estimate
            the distribution
            n (int): number of Bernoulli trials
            p (float): probability of a "success"
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # calculate the lambtha value
            self.n = len(data)
            self.p = float(sum(data) / len(data))
        else:
            # If data is not given
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
