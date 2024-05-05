#!/usr/bin/env python3
"""Generate a bar graph with stacked bars."""


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """This function generates a stacked bar graph displaying
    the quantity of fruits per person."""
    np.random.seed(5)

    fruit_quantities = np.random.randint(0, 20, (4, 3))

    plt.figure(figsize=(6.4, 4.8))

    bar_width = 0.5
    positions = [1, 2, 3]
    labels = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    fruit_names = ["apples", "bananas", "oranges", "peaches"]

    for i, fruit in enumerate(fruit_names):
        plt.bar(
            positions,
            fruit_quantities[i],
            color=colors[i],
            label=fruit,
            width=bar_width,
            bottom=np.sum(fruit_quantities[:i], axis=0) if i > 0 else 0
        )

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")

    plt.xticks(positions, labels)
    plt.legend()

    # Display the plot
    plt.show()
bars()