#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    bar_width = 0.5
    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    
    for i in range(4):
        plt.bar(names, fruit[i], bar_width, color=colors[i],
                bottom=np.sum(fruit[:i], axis=0), label=fruit_names[i])

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 80, 10))
    plt.title('Number of Fruit per Person')

    plt.legend()

    plt.show()
