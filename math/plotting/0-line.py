#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)  # Define x values from 0 to 10

plt.figure()          # Start a new figure
plt.plot(x, y, '-r')  # Plot y against x with a red solid line
plt.xlim(0, 10)       # Set x-axis limits
plt.show()            # Show the plot
