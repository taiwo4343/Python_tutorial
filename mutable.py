'''
This script demonstrates some mutable/immutable differences, and functions
'''

import numpy as np

# An important "gotcha" in python involves mutable/immutable data types

# immutable data types cannot be edited, only overwritten or copied, so they are
#   simple to work with.
my_tuple = (1,2,3) # tuples are immutable collection of objects
my_tuple_cpy = my_tuple # a true copy is made
my_tuple = (4,5,6) # this overwrites my_tuple, but not my_tuple_cpy
print('my_tuple = {}'.format(my_tuple))
print('my_tuple_cpy = {}'.format(my_tuple_cpy))
print('---------------------------')

# actually, the above will work the same way with lists (mutable), because
#   a new object is created when my_tuple is reassigned:
my_list = [1,2,3] # mutable
my_list_ref = my_list # this now refers to the same object in memory as my_list
my_list = [4,5,6] # a new object is created, and my_list refers to it.
                  #   but my_list_ref still refers to [1,2,3]
print('my_list = {}'.format(my_list))
print('my_list_ref = {}'.format(my_list_ref))
print('---------------------------')

# BUT!!!
my_list = [1,2,3]
my_list_ref = my_list # both my_list and my_list_ref point to the same object
my_list[0] = 0 # so when I alter my_list, my_list_ref is altered too!
print('my_list = {}'.format(my_list))
print('my_list_ref = {}'.format(my_list_ref))
print('---------------------------')

# If you don't want this, you can force a "true copy" to be made as follows:
my_list = [1,2,3]
my_list_cpy = list(my_list) # the list function makes a true copy
my_list[0] = 0 # so when I alter my_list, my_list_cpy remains the same
print('my_list = {}'.format(my_list))
print('my_list_cpy = {}'.format(my_list_cpy))
print('---------------------------')

# Remember that NumPy ndarrays are mutable, so you have to worry about this often.

#####################################

# This is how you create a function:
def my_function(array1, array2):
    '''
    This is the doc string for the function. Good doc strings say briefly what
    the purpose of the function is, what it takes in, and what it spits out.

    Arguments:
        array1: 2D ndarray
        array2: 2D ndarray

    Returns:
        ndarray
    '''

    # When mutable data types are passed into a function, they are passed by
    #   reference. That means that the actual array is passed in, not a copy - 
    #   if you alter the array in the function, it is altered outside the function!
    #   This is faster than creating a true copy, but can result in hard to debug
    #   errors, especially if you alter the array by accident.

    array1[0,0] = 0
    array2_cpy = np.array(array2) # remember you need a true copy!
    array2_cpy = array2_cpy * 5

    return array2_cpy

# Now we test out the function...
A = np.eye(3) # 3x3 identity matrix
B = np.ones((3,3)) # 3x3 array of ones

print('A before function:')
print(A)
print('B before function:')
print(B)
print('------------------')

C = my_function(A, B)

print('A after function:')
print(A)
print('B after function:')
print(B)
print('Array returned by function:')
print(C)


# There are other, somewhat more complicated "gotchas" in Python, but not very many.
# Google "Python gotchas" and read about them!