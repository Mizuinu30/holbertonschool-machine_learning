#!/usr/bin/env python3
"""Generate a bar graph with stacked bars."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Generate a bar graph with stacked bars."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    labels = ["apples", "bananas", "oranges", "peaches"]

    for i in range(4):
        plt.bar(
            [1, 2, 3],
            fruit[i],
            color=colors[i],
            bottom=np.sum(fruit[:i], axis=0),
            label=labels[i],
            width=0.5,
        )

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")
    plt.xticks([1, 2, 3], names)
    plt.legend()
    plt.show()


bars()