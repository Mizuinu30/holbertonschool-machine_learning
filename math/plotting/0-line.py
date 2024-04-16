#!/usr/bin/env python3
"""A function that plots a red line"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """A function that plots a red line"""
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)

    plt.figure()
    plt.plot(x, y, '-r')
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":
    line()
