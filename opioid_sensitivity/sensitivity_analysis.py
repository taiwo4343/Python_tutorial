'''Use SALib to perform Sobol sensitivity analysis on the opioid model parameters.
This file is capable of reproducing Fig. 4 in Battista, Pearcy, and Strickland (2019).
'''

import sys, os, time
import pickle # use to temporarily save the results
import argparse
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import opioid_model

default_N = os.cpu_count()
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="obtain N*(2D+2) samples from parameter space")
parser.add_argument("-n", "--ncores", type=int,
                    help="number of cores, defaults to {}".format(default_N))
parser.add_argument("-o", "--filename", type=str, 
                    help="filename to write output to, no extension",
                    default='analysis')



def run_full_model(alpha,beta_A,beta_P,delta,epsilon,gamma,zeta,mu,mu_star,sigma,nu_1=0,nu_2=0):
    '''Defines a model wrapper based on the parameter space in main().
    It should take in as arguments all the parameters defined in the SALib
    problem definition.'''
    # When to start/end the model run
    tstart = 0
    tstop = 10
    #tstop =10000
    # Copy default parameter dict to have something to work with
    params = dict(opioid_model.params)
    # Replace other parameter values
    params['alpha'] = alpha
    params['beta_A'] = beta_A
    params['beta_P'] = beta_P
    params['delta'] = delta
    params['epsilon'] = epsilon
    params['gamma'] = gamma
    params['zeta'] = zeta
    params['nu_1'] = nu_1
    params['nu_2'] = nu_2
    params['mu'] = mu
    params['mu_star'] = mu_star
    params['sigma'] = sigma
    # Fix initial conditions based on data
    P_0 = 0.05 #Study: Boudreau et al +  time increase
    A_0 = 0.0062 #SAMHSA: https://www.samhsa.gov/data/sites/default/files/NSDUH-FFR2-2015/NSDUH-FFR2-2015.pdf
    R_0 = 0.0003 #HSS Treatment Episode Data Set https://www.samhsa.gov/data/sites/default/files/2014_Treatment_Episode_Data_Set_National_Admissions_9_19_16.pdf
    S_0 = 1 - P_0 - A_0 - R_0
    # Run model
    try:
        result = opioid_model.solve_odes(S_0,P_0,A_0,R_0,tstart,tstop,params)
    except:
        # In case of failure, return info about the failure and make it easy
        #   to find in the data.
        return (sys.exc_info()[1],None,None,None)
    # Return just the end value of each variable (S, P, A, R)
    return (result[0][-1], result[1][-1], result[2][-1], result[3][-1])



def main(N, filename, pool=None):
    '''Runs parameter sensitivity on the opioid model'''

    ### Define the parameter space ###
    problem = {
        'num_vars': 12, #number of parameters
        'names': ['alpha', 'beta_A', 'beta_P', 'delta', 'epsilon', 'gamma',
                    'zeta', 'mu', 'mu_star', 'sigma','nu_1','nu_2'],
        'bounds': [[0.02,0.2], [0.0001,0.01], [0.0001,0.01], [0,1], [0.8,8], [0.00235,0.0235],
                    [0.2,2], [0.0023,0.023], [0.00365,0.0365], [0,1], [0,1], [0,1]]
    }

    ### Create an N*(2D+2) by num_var matrix of parameter values ###
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    ### Run model ###
    print('Examining the parameter space.')
    if args.ncores is None:
        poolsize = os.cpu_count()
    else:
        poolsize = args.ncores
    chunksize = param_values.shape[0]//poolsize
    output = pool.starmap(run_full_model, param_values, chunksize=chunksize)

    ### Parse and save the output ###
    print('Saving and reviewing the results...')
    param_values = pd.DataFrame(param_values, columns=problem['names'])
    # write data to temporary location in case of errors
    with open("raw_result_data.pickle", "wb") as f:
        result = {'output':output, 'param_values':param_values}
        pickle.dump(result, f)
    # Look for errors
    error_num = 0
    error_places = []
    for n, result in enumerate(output):
        if result[1] is None:
            error_num += 1
            error_places.append(n)
    if error_num > 0:
        print("Errors discovered in output.")
        print("Parameter locations: {}".format(error_places))
        print("Please review pickled output.")
        return
    # Save results in HDF5 as dataframe
    print('Parsing the results...')
    output = np.array(output)
    # Resave as dataframe in hdf5
    store = pd.HDFStore(filename+'.h5')
    store['param_values'] = param_values
    store['raw_output'] = pd.DataFrame(output, columns=['S', 'P', 'A', 'R'])
    os.remove('raw_result_data.pickle')

    ### Analyze the results and view using Pandas ###
    # Conduct the sobol analysis and pop out the S2 results to a dict
    S2 = {}
    S_sens = sobol.analyze(problem, output[:,0], calc_second_order=True)
    S2['S'] = pd.DataFrame(S_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['S_conf'] = pd.DataFrame(S_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    P_sens = sobol.analyze(problem, output[:,1], calc_second_order=True)
    S2['P'] = pd.DataFrame(P_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['P_conf'] = pd.DataFrame(P_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    A_sens = sobol.analyze(problem, output[:,2], calc_second_order=True)
    S2['A'] = pd.DataFrame(A_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['A_conf'] = pd.DataFrame(A_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    R_sens = sobol.analyze(problem, output[:,3], calc_second_order=True)
    S2['R'] = pd.DataFrame(R_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['R_conf'] = pd.DataFrame(R_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    # Convert the rest to a pandas dataframe
    S_sens = pd.DataFrame(S_sens,index=problem['names'])
    P_sens = pd.DataFrame(P_sens,index=problem['names'])
    A_sens = pd.DataFrame(A_sens,index=problem['names'])
    R_sens = pd.DataFrame(R_sens,index=problem['names'])

    ### Save the analysis ###
    print('Saving...')
    store['S_sens'] = S_sens
    store['P_sens'] = P_sens
    store['A_sens'] = A_sens
    store['R_sens'] = R_sens
    for key in S2.keys():
        store['S2/'+key] = S2[key]
    store.close()

    # Plot
    plot_S1_ST(S_sens, P_sens, A_sens, R_sens, True)



def load_data(filename):
    '''Load analysis data from previous run and return for examination
    (e.g. in iPython). This function will return a Pandas store HDF5 object.'''

    return pd.HDFStore(filename)



def plot_S1_ST_from_store(store, show=True):
    '''Extract and plot S1 and ST sensitivity data directly from a store object'''

    plot_S1_ST(store['S_sens'], store['P_sens'], store['A_sens'],
               store['R_sens'], show)



def print_max_conf(store):
    '''Print off the max confidence interval for each variable in the store,
    for both first-order and total-order indices'''
    for var in ['S_sens', 'P_sens', 'A_sens', 'R_sens']:
        print('----------- '+var+' -----------')
        print('S1_conf_max: {}'.format(store[var]['S1_conf'].max()))
        print('ST_conf_max: {}'.format(store[var]['ST_conf'].max()))
        print(' ')



def plot_S1_ST(S_sens, P_sens, A_sens, R_sens, show=True):
    # Gather the S1 and ST results
    S1 = pd.concat([S_sens['S1'], P_sens['S1'], A_sens['S1'], 
                   R_sens['S1']], keys=['S','P','A','R'], axis=1) #produces copy
    ST = pd.concat([S_sens['ST'], P_sens['ST'], A_sens['ST'], 
                   R_sens['ST']], keys=['S','P','A','R'], axis=1)
    # Change to greek
    for id in S1.index:
        if id != 'mu_star':
            S1.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
        else:
            S1.rename(index={id: r'$\mu^*$'}, inplace=True)
    for id in ST.index:
        if id != 'mu_star':
            ST.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
        else:
            ST.rename(index={id: r'$\mu^*$'}, inplace=True)
    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    # Switch the last two colors so A is red
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:4]
    clr = colors[-1]; colors[-1] = colors[-2]; colors[-2] = clr
    S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8, color=colors)
    ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, color=colors)
    for ax in axes:
        ax.tick_params(labelsize=18)
        ax.legend(fontsize=16)
    axes[0].set_title('First-order indices', fontsize=26)
    axes[1].set_title('Total-order indices', fontsize=26)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.pdf".format(time.strftime("%m_%d_%H%M")))
    return (fig, axes)



def plot_S1_ST_tbl_from_store(store, show=True, ext='pdf'):
    S_sens = store['S_sens']
    P_sens = store['P_sens']
    A_sens = store['A_sens']
    R_sens = store['R_sens']
    # Gather the S1 and ST results
    S1 = pd.concat([S_sens['S1'], P_sens['S1'], A_sens['S1'], 
                   R_sens['S1']], keys=['S','P','A','R'], axis=1) #produces copy
    ST = pd.concat([S_sens['ST'], P_sens['ST'], A_sens['ST'], 
                   R_sens['ST']], keys=['S','P','A','R'], axis=1)
    # Change to greek
    for id in S1.index:
        if id != 'mu_star':
            S1.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
        else:
            S1.rename(index={id: r'$\mu^*$'}, inplace=True)
    for id in ST.index:
        if id != 'mu_star':
            ST.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
        else:
            ST.rename(index={id: r'$\mu^*$'}, inplace=True)
    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.1,2.1,1.1])
    axes = []
    for ii in range(2):
        axes.append(plt.subplot(gs[ii]))
    # Switch the last two colors so A is red
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:4]
    clr = colors[-1]; colors[-1] = colors[-2]; colors[-2] = clr
    bar_S1 = S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8, color=colors)
    bar_S2 = ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, color=colors)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        #ax.get_yaxis().set_visible(False)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom=0)
    axes[0].set_title('First-order indices', fontsize=26)
    axes[1].set_title('Total-order indices', fontsize=26)
    # Create table
    columns = ('Value Range',)
    rows = list(S1.index)

    ##### Text to print in table alongside plot #####
    # These ranges need to match those used in the problem definition!!
    cell_text = [['.02-.2'], ['0.0001-0.01'], ['0.0001-0.01'], ['0-1'], ['.8-8'], ['.00235-.0235'],
                 ['.2-2'], ['.0023-.023'], ['.00365-.0365'], ['0-1'], ['0-1'], ['0-1']]

    tbl_ax = plt.subplot(gs[2])
    the_table = tbl_ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                 loc='center')
    the_table.set_fontsize(18)
    the_table.scale(1,2.3)
    the_table.auto_set_column_width(0)
    tbl_ax.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.{}".format(time.strftime("%m_%d_%H%M"), ext))
    return (fig, axes)



if __name__ == "__main__":
    args = parser.parse_args()
    if args.ncores is None:
        with Pool() as pool:
            main(args.N, filename=args.filename, pool=pool)
    else:
        with Pool(args.ncores) as pool:
            main(args.N, filename=args.filename, pool=pool)

