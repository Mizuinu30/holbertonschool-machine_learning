#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']
fruits = {
    'apples': 'red',
    'bananas': 'yellow',
    'oranges': '#ff8000',
    'peaches': '#ffe5b4'
}
qty = len(people)
i = 0
for name, color in sorted(fruits.items()):
    bottom = 0
    for i2 in range(i):
        bottom += fruit[i2]
    plt.bar(
        np.arange(qty),
        fruit[i],
        width=0.5,
        bottom=bottom,
        color=color,
        label=name
    )
    i += 1
plt.xticks(np.arange(qty), people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
