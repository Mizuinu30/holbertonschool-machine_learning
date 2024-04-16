#!/usr/bin/env python3
""" Frequency of student grades """
import numpy as np
import matplotlib.pyplot as plt



def frequency():
    """ Frequency of student grades"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=10, range=(0, 100), rwidth=10,
             edgecolor='black')
    plt.show()

frequency()
