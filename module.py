'''
An example of a Python module, which contains various functions and things
'''

# Some variables

alpha = 4
beta = 7.2

cheeses = ['brie', 'cheddar', 'roquefort', 'morbier']

endless = ('Death', 'Dream', 'Destruction', 'Despair', 'Desire', 'Destiny', 'Delirium')

def mult_them(A, B):
    ''' Tries to multiply two objects. If it fails, print a message and
    return None.'''

    try:
        result = A*B
        return result
    except:
        print("Couldn't do it!!!")
        return None # special null-type object



def get_endless(index=1):
    '''Here, I'm referencing a variable outside of the function definition... Python
    will always look inside the function for the variable first, and then go
    outside. Be careful with this... it can lead to unintended consquences!

    The default index is 1.
    '''

    try:
        return endless[index]
    except IndexError: # It's good to be specific about the type of error you are catching!
        print('No such index.')

    