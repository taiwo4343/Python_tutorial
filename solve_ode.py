'''Example of solving an ODE with scipy'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def logistic_eqn(t, x, params):
    '''This function specifies the ODE(s) to be solved. It must take in the
    current time as the first argument and the current value(s) of the
    independent variables as the second argument. You can also pass in parameters
    with a third argument. I like to use a dictionary for this, as I can give
    the parameters names and don't have to keep track of the order in which I
    put them into a list or somesuch.'''
    
    dxdt = params['r']*x*(1-x/params['K']) # logistic equation

    return dxdt



def solve_logistic(x0, tstart, tstop, params):
    '''Solve the logistic equation with given initial condition, start time,
    end time, and parameters.'''

    y0 = np.array([x0]) # initial condition must be given as an array

    tmesh = np.linspace(tstart,tstop,1000) # force recorded solutions at these points.
    # this can be passed into the t_eval argument of solve_ivp.

    ### call the solver ###
    # The default solver is RK45 which is an explicit, variable step size Runge
    #   Kutta solver of order 4(5). It's also known as "Dormand-Prince" and 
    #   it's basically ODE45. Try it first unless you have a good reason not to.
    # args passes arguments to all supplied functions. It must be a tuple.
    solution = solve_ivp(logistic_eqn, t_span=[tstart, tstop], y0=y0, args=(params,))

    # After we've finished, return the solution object.
    return solution



def plot_solution(sol):
    '''Plot a solution set and either show it or return the plot object'''
    time = sol.t
    x = sol.y[0,:] # only one eqn., so has shape (1, timepoints)
    plt.plot(time, x, label='logistic eqn')
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.show()



# this if statement is a bit of boilerplate that says:
#   "If this script is run from a command prompt, do the following"
#   This allows a script to have bits that can be used by other scripts,
#   as well as bits that will only be run if the script is directly called.
if __name__ == "__main__":
    x0 = 1 # Initial condition for x

    params = {} # this creates a dictionary
    params['r'] = 1
    params['K'] = 1000

    tstart = 0 # start solving the ODE at time t=0
    tstop = 60 # stop at time t=100

    # Run the solver
    sol = solve_logistic(x0, tstart, tstop, params)
    plot_solution(sol)