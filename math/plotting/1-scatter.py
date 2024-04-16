#!/usr/bin/env python3
"""A function that creates a scatter plot"""


import numpy as np
import matplotlib.pyplot as plt

def scatter():
    """ Crestes a scatter plot"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.figure(figsize=(8, 6))  # Adjusting figure size to be more suitable for comparison
    plt.scatter(x, y, color='magenta', s=10)  # s is the size of the points, set to a standard value for visibility
    plt.title("Men's Height vs Weight")  # Title of the plot
    plt.xlabel('Height (in)')  # x-axis label
    plt.ylabel('Weight (lbs)')  # y-axis label
    plt.xlim(30, 110)  # Setting x limits to show all data points clearly
    plt.ylim(100, 260)  # Setting y limits to show all data points clearly
    plt.show()
    
scatter()
   