#!/usr/bin/env python3
""" A Function that crates a histogram"""
import numpy as np
import matplotlib.pyplot as plt

def frequency():
    """Function that crates a histogram"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    bin_size = 10
    bins = np.arange(0, 101, bin_size)
    
    plt.figure(figsize=(6.4, 4.8))
    
    # Use blue color for bars and align them to the left of bin edges
    plt.hist(student_grades, bins=bins, color='blue', edgecolor='black', align='left')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    
    plt.xticks(bins)
    
    # Remove yticks if they are not required by the reference image
    # plt.yticks([])

    plt.xlim([0, 100])  # Ensure the x-axis starts at 0 and ends at 100
    
    plt.show()

frequency()
