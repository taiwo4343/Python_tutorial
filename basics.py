'''
Three quotations begins/ends a multi-line comment to be used for documenation 
(also known as a doc string).
Include one of these at the beginning of every py file to give a brief description
of what the code contains, who wrote it, etc. Also include one of these at the 
beginning of every function definition to explain what the function does.

This script shows some very basic parts of Python. Run it using the command
"python basics.py" in a terminal.

Created on Tues Jun 4 2019

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

# This is a single line comment. Use it very often.
# You will save yourself untold amounts of grief if you remember these words:
#   "We write code to be read by humans, not machines."

# Place all imports at the top of your script. Import only the stuff you need,
#   and do not import *. It clutters the namespace and can lead to hard to debug
#   errors.

import numpy as np

# Lets do some basic things...

spam = 'Have some spam.' # This is a string. It's an immutable type.
eggs = "Have some eggs." # Immutable means you can't alter it in place, 
                         #  only completely overwrite it.
                         # You can use double or single quotes for strings

print(spam) # Prints the string to the terminal and starts a new line.
print(eggs)

# Remember that Python is base 0.
# All intervals in Python include the left endpoint and exclude the right one.

print(eggs[0:10]) # Will print characters 0-9 in the eggs string (10 characters total)

for ii in range(3): # ii will take on values 0, 1, 2.
    # print characters starting at 5 from the end and going all the way to the end
    print(eggs[-5:])

# for loops finish when you quit indenting
print(spam[0:5] + spam[5:]) # addition concatenates strings

mylist = [] # This is an empty list. Lists are the most basic mutable type, and 
            #   are what you should use anytime you will need to add or remove things.
            #   They can hold any combination of any type of object, including
            #   other lists.
mylist.append('to wong foo') # this appends a string
mylist.append([3,5,7]) # this appends a list of numbers
mylist.append({'pirate': 'arggggg'}) # this appends a dictionary
mylist.append(49) # this appends a number
print(mylist)
# You can use the format property of a string to replace {} with text, etc. at runtime.
pirate_call = 'The pirate says {}!!!'.format(mylist[2]['pirate'])
print(pirate_call)
print(' ') # add an empty line

# Here are some more examples of flow control:
done_flag = False
while not done_flag:
    your_number_txt = input('Input an odd number greater than 6: ') # get some user input
    # input returns a string. turn it into an integer
    your_number = int(your_number_txt)
    if your_number > 6 and your_number % 2 == 0: # % is the modulus operator
        print('Happy land!')
        done_flag = True # this will cause us to break out of the while loop
    elif your_number > 6 and not your_number % 2 == 0:
        print('The number {} is not even.'.format(your_number))
    else:
        # Note that to get the word It's, I use double quotations around the string
        print("{} is a silly number. It's just not large enough!".format(your_number))

print(' ') # add an empty line after the prompt stuff

# (Nearly) everything in NumPy uses the array datatype.
# It is mutable, but has a fixed shape and should only contain a single datatype
#   (usually numbers, and most usually floating point numbers)
# You can create an array out of a list, or a list of lists.
A = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3x3 array of integers
print('A = ')
print(A)
B = np.random.rand(3,3) # 3x3 array of floating point numbers between 0 and 1
print('B = ')
print(B)
# All operations are elementwise...
print('Element-wise multiplication:')
print(A*B)
# In Python 3.5+, we also have @ which is matrix multiplication
print('Matrix multiplicaton:')
print(A@B)
# Tons of other routines can be found in the numpy library
# You can get a particular element from a multidimensional array like this:
print('Lucky number {}!'.format(A[2,0])) # row 2, column 0
# Slices (with :) work too, and there are lots of other fancy tricks!
