#!/usr/bin/env python3
""" Function that creates a plot bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Function that creates a plot bars"""
np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


apples = fruit[0].copy()
bananas = fruit[1].copy()
oranges = fruit[2].copy()
peaches = fruit[3].copy()

bars = np.add(apples, bananas).tolist()
bars2 = np.add(bars, oranges).tolist()
r = [0, 1, 2]

names = ['Farrah', 'Fred', 'Felicia']
width = 0.5

plt.bar(r, apples, width, color='red', label='apples')
plt.bar(r, bananas, width, bottom=apples, color='yellow', label='bananas')
plt.bar(r, oranges, width, bottom=bars, color='#ff8000', label='oranges')
plt.bar(r, peaches, width, bottom=bars2, color='#ffe5b4', label='peaches')

plt.yticks(range(0, 90, 10))
plt.xticks(r, names)
plt.ylabel('Quantity of Fruit')
plt.suptitle('Number of Fruit per Person')
plt.legend()

plt.show()
