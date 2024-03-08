'''
Build up to homework 1
'''
import numpy as np

# We have the fecundity rates and survival probabilities
F = np.array([0, 0, 0, 5, 2, 0])
P = np.array([0.4, 0.6, 0.7, 0.9, 0.8, 0])

# let us Create a 6x6 Leslie matrix
L = np.zeros((6, 6))
L[0, :] = F  # The aim is to fill the first row with fecundity rates
for i in range(1, 6):
    L[i, i-1] = P[i-1]  # Then we fill the sub-diagonal with survival probabilities
