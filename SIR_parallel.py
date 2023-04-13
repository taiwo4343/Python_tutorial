'''
Solve the SIR equations in parallel while varying parameters
'''

from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#time points to solve at
tpts = np.linspace(0,200,1001) #run longer

#initial values as population fractions
I0 = 1e-2
R0 = 0

# parameter values
params = {}
params['beta'] = 1.4247
params['gamma'] = 0.14286
params['mu'] = 0.01

beta_list = np.linspace(0,4,10001) # way too many points.

##################################

# vectorize initial conditions
x0 = np.array([1-I0-R0, I0, R0])

# define ode equations
def SIR_ODEs_beta(t,x,params,beta):
    '''This function returns the time derivates of S,I,R.

    The ode solver expects the first two arguments to be t and x
    NOTE: This is the OPPPOSITE order from scipy.integrate.odeint!!

    The params argument should be a dict with beta, gamma, and mu as keys.
    It must be passed into the solver using the set_f_params method
    '''

    S = x[0]; I = x[1]; R = x[2]
    dx = np.zeros(3)

    dx[0] = -beta*S*I + params['mu']*(I+R)
    dx[1] = beta*S*I - params['gamma']*I - params['mu']*I
    dx[2] = params['gamma']*I - params['mu']*R

    return dx



# Create a short function that solves the ODEs given a beta and returns
#   only the final values.
def get_final_value(beta):
    sol = solve_ivp(SIR_ODEs_beta, t_span=[tpts[0], tpts[-1]], y0=x0, 
                    args=(params,beta))
    return sol.y[:,-1]



def main(pool):
    ##### Solve procedure in parallel #####
   
    # apply our new function to all of our beta values in parallel
    results = pool.map(get_final_value, beta_list)

    ##### Plot result #####
    fig = plt.figure(figsize=(9,7))
    fsol = np.array(results)
    plt.plot(beta_list,fsol[:,0],beta_list,fsol[:,1],beta_list,fsol[:,2])
    plt.legend(['S-final','I-final','R-final'])
    plt.title("Plot of $S,I,R$ final by $\\beta$")
    plt.xlabel("$\\beta$")
    plt.ylabel("population fraction")
    plt.show()



if __name__ == "__main__":
    with Pool(processes=4) as peaches:
        main(peaches)
