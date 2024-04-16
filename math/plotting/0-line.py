#!/usr/bin/env python3
"""A function that plots a red line"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """A function that plots a red line"""
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)  # x values from 0 to 10

    plt.figure()           # Start a new figure
    plt.plot(x, y, '-r')   # Plot y against x with a red solid line
    plt.xlim(0, 10)        # Set x-axis limits from 0 to 10
    plt.show()             # Display the plot


if __name__ == "__main__":
    line()