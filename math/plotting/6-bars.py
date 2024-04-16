#!/usr/bin/env python3
""" Function that creates a plot for bars"""

import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    people = ['Farrah', 'Fred', 'Felicia']
    bar_positions = np.arange(len(people))
    bottoms = np.zeros(len(people))

    for i, (name, color) in enumerate(zip(fruit_names, colors)):
        plt.bar(bar_positions, fruit[i, :], color=color, width=0.5, bottom=bottoms, label=name)
        bottoms += fruit[i, :]

    plt.xticks(bar_positions, people)
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    bars()
