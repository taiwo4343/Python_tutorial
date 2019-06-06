'''Example of solving an ODE with scipy'''

import numpy as np
from scipy.integrate import ode
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

    x_sol = [x0] # create a list to hold the solution. it starts with the IC
    time = [tstart] # create a list to hold the times which correspond to the solution points

    dt = 0.1 # record solutions at this time interval

    ### setup solver ###
    # We will use a "Dormand-Prince" numerical solver, which is a 4th-order 
    #   Runge-Kutta method with variable stepsize. It's basically ODE45.
    solver = ode(logistic_eqn).set_integrator('dopri5')
    # Pass in the IC, initial time, and parameter dictionary to the solver object
    solver.set_initial_value(x0, tstart).set_f_params(params)
    # Solve. This involves a loop in which we integrate to each new time we want
    #   to record, and then record it.
    # The nice thing about this approach is that we can do complicated things, like
    #   test properties of the solution as it is coming out and take action accordingly.
    while solver.successful and solver.t < tstop:
        solver.integrate(solver.t+dt) #integrate up to next time we want to record
        # record solution at that time
        x_sol.append(solver.y)
        time.append(solver.t)

    # After we've finished, return all the values for plotting, recording, etc.
    return (x_sol, time)



def plot_solution(x, time):
    '''Plot a solution set and either show it or return the plot object'''
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
    x, time = solve_logistic(x0, tstart, tstop, params)
    plot_solution(x, time)