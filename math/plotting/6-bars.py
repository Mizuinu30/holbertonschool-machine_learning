#!/usr/bin/env python3
"""Generate a bar graph with stacked bars."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Generate a bar graph with stacked bars."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    plt.bar([1, 2, 3], fruit[0], color="red", label="apples", width=0.5)
    plt.bar(
        [1, 2, 3], fruit[1], color="yellow", bottom=fruit[0],
        label="bananas", width=0.5
    )
    plt.bar(
        [1, 2, 3],
        fruit[2],
        color="#ff8000",
        bottom=fruit[0] + fruit[1],
        label="oranges",
        width=0.5,
    )
    plt.bar(
        [1, 2, 3],
        fruit[3],
        color="#ffe5b4",
        bottom=fruit[0] + fruit[1] + fruit[2],
        label="peaches",
        width=0.5,
    )
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.title("Number of Fruit per Person")

    plt.xticks([1, 2, 3], ["Farrah", "Fred", "Felicia"])
    plt.legend()
    plt.show()
