#!/usr/bin/env python3
"""This modlue contains the poisson distribution class
"""


class Poisson:
    """ Class Poisson that represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """ Constructor of the Poisson class
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        return (self.lambtha ** k) * (2.7182818285 ** (
            -1 * self.lambtha)) / factorial_k

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
