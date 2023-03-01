'''
Show basic plotting using matplotlib
'''

import numpy as np
import matplotlib.pyplot as plt

x_vals = np.arange(-10,10,0.1)

y_vals = x_vals**2 - 10 # A simple quadratic

plt.plot(x_vals, y_vals) # line plot
plt.title('Parabola')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.show() # must do this to show the plot! program will pause here until the
           #    plot is dismissed.

# something a little more complicated...
with plt.xkcd(): # "with" creates a context for a command. this is just for fun!
    plt.figure(figsize=(6, 4)) # custom figure size
    plt.subplot(211) # two rows, one column, plot number one
    plt.plot(x_vals, x_vals, label='linear plot') # linear
    plt.plot(x_vals, np.exp(x_vals), label='exp plot') # exponential
    plt.legend() # make a legend
    plt.title('Linear vs. exponential')
    plt.subplot(212) # two rows, one column, plot number two
    plt.scatter(x_vals, x_vals + np.random.rand(len(x_vals))) # linear plus some noise
    plt.title('Noisy linear')
    plt.tight_layout() # give some space so the title doesn't overlap things
    plt.show()