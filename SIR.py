"""
Solves the SIR system of ODEs
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#time points to solve at
tpts = np.linspace(0,100,1001)

#initial values as population fractions
I0 = 1e-2
R0 = 0

# parameter values
params = {}
params['beta'] = 1.4247
params['gamma'] = 0.14286
params['mu'] = 0.01


##################################

# vectorize initial conditions
x0 = np.array([1-I0-R0, I0, R0])

# define ode equations
def SIR_ODEs(t,x,params):
    '''This function returns the time derivates of S,I,R.

    The ode solver expects the first two arguments to be t and x
    NOTE: This is the OPPPOSITE order from scipy.integrate.odeint!!

    The params argument should be a dict with beta, gamma, and mu as keys.
    It must be passed into the solver using the set_f_params method
    '''

    S = x[0]; I = x[1]; R = x[2]
    dx = np.zeros(3)

    dx[0] = -params['beta']*S*I + params['mu']*(I+R)
    dx[1] = params['beta']*S*I - params['gamma']*I - params['mu']*I
    dx[2] = params['gamma']*I - params['mu']*R

    return dx

##### Solve procedure #####
sol = solve_ivp(SIR_ODEs, t_span=[tpts[0], tpts[-1]], y0=x0, t_eval=tpts, 
                args=(params,))

##### Plot result #####
fig = plt.figure(figsize=(9,7))
plt.plot(sol.t,sol.y[0,:],sol.t,sol.y[1,:],sol.t,sol.y[2,:])
plt.legend(['S','I','R'])
plt.title("Plot of $S,I,R$ vs. time")
plt.xlabel("$t$")
plt.ylabel("population fraction")
plt.show()

