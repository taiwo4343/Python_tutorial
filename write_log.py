'''Basic example of writing a log file.'''

import sys, datetime
import numpy as np

with open("log_ex_" + datetime.datetime.now().strftime("%m_%d_%H_%M") + '.log', 'w') as sys.stdout:
    # Just output a bunch of fake data.
    for n in range(100):
        print("Output line:", n)
        print("Data list:", np.random.rand(4))
        sys.stdout.flush() # This flushes the buffer to file
