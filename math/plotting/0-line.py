#!/usr/bin/env python3
"""Function that plots a line in a graph"""


import numpy as np
import matplotlib.pyplot as plt


def line():
    y = np.arange(0, 11) ** 3
    
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.xlim(0, len(y)-1)
    
    plt.show()

line()
