'''
Run Sobol sensitivity analysis for the opioid model, utilizing Saltelli's sequence.
This uses the SALib library and draws on Python multiprocessing to conduct
simulations in parallel. The resulting data is handled by Pandas.

install SALib with "conda install -c conda-forge salib"
You will also need to install pyqt and pytables.

This script calls and runs sensitivity analysis on opioid_model.py

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import sys, os, time, warnings
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import pickle
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
                    help="number of cores, defaults to {} on this machine".format(default_N))
parser.add_argument("-o", "--filename", type=str, 
                    help="filename to write output to, no extension",
                    default='analysis')

def run_model(alpha, beta_P, beta_A, delta, epsilon, gamma, zeta):
    '''This is a wrapper around the py-file that solves the model we are
    conducting sensitivity analysis on. It takes in each parameter we are
    testing as a separate argument and calls the solver, getting a result back.
    It then parses the result into whatever metric we are conducting sensitivity
    on, and returns it.
    '''

    # Copy default parameter dict
    params = dict(opioid_model.params)
    # Replace parameter values in this dict that we are varying
    params['alpha'] = alpha
    params['beta_P'] = beta_P
    params['beta_A'] = beta_A
    params['delta'] = delta
    params['epsilon'] = epsilon
    params['gamma'] = gamma
    params['zeta'] = zeta

    # Run model
    try:
        S,P,A,R = opioid_model.solve_odes(p=params)
    except:
        # This is to make debugging easier.
        # Instead of crashing when a parameter set fails to work, we save
        #   None for it's output, along with the error message in sys.exc_info.
        #   That way, we can search for this failure later in our data, and
        #   figure out which parameter combinations are throwing the error.
        return (sys.exc_info()[1], None, None, None, None)

    ##### Convert model solution into scalar values w.r.t which we measure sensitivity #####
    if not isinstance(S, Exception):
        # In this case, we will just measure sensitivity to the end values of
        #   S,P,A,R after a set simulation length
        return (S[-1], P[-1], A[-1], R[-1])
    else:
        # An error was thrown, return as-is for later processing
        return (S,P,A,R)



def main(N, filename, ncores=None, pool=None):
    '''Runs parameter sensitivity on the model.

    Arguments:
        N: int, used to calculate the number of model realizations to sample
        filename: output file name for plots and data
        ncores: number of cores we are running on (for determining chunk size)
        pool: multiprocessing pool
    '''

    #############################################################################
    ### Define the parameter space within the context of a problem dictionary ###
    #############################################################################

    problem = {
        # number of parameters
        'num_vars' : 7,
        # parameter names. order matters.
        'names' : ['alpha', 'beta_P', 'beta_A', 'delta',
                   'epsilon', 'gamma', 'zeta'], 
        # bounds for each corresponding parameter. order matters
        'bounds' : [[0.02,0.2], [0.0001,0.01], [0.0001,0.01], [0,1], 
                    [0.8,8], [0.00235,0.0235], [0.2,2]]
    }

    # name of the output variables that we are measuring sensitivity to
    observable_names = ['S', 'P', 'A', 'R']

    ###############################################
    ###             Sample and solve            ###
    ###############################################

    ### Create an N*(2D+2) by num_vars matrix of parameter values to sample ###
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    ### Run model ###
    print('Examining the parameter space.')
    if pool is not None:
        if ncores is None:
            poolsize = os.cpu_count()
        else:
            poolsize = ncores
        chunksize = param_values.shape[0]//poolsize
        output = pool.starmap(run_model, param_values, chunksize=chunksize)
    else:
        # Usually a bad idea to run in serial. Say so.
        print('Warning!!! Running in serial only! Multiprocessing pool not utilized.')
        output = []
        for params in param_values:
            output.append(run_model(*params))

    ##########################################################
    ###         Look for errors and save the output        ###
    ##########################################################

    print('Saving and checking the results for errors...')
    # first, dump data to temporary location in case anything obnoxious happens
    with open("raw_result_data.pickle", "wb") as f:
        result = {'output':output, 'param_values':param_values}
        pickle.dump(result, f)

    ### Look for errors ###
    error_num = 0
    error_places = []
    for n, result in enumerate(output):
        if isinstance(result[0], Exception):
            error_num += 1
            error_places.append(n)
    if error_num > 0:
        print("Errors discovered in output.")
        print("Parameter locations: {}".format(error_places))
        # remove the errors from the dataset and store them in a pickle file
        #   for later examination
        print("Pickling errors...")
        err_output = []
        err_params = []
        for idx in error_places:
            err_output.append(output.pop(idx)) # pull out err output
            err_params.append(param_values[idx,:]) # reference err param values
        with open("err_results.pickle", "wb") as f:
            err_result = {'err_output':err_output, 'err_param_values':param_values}
            pickle.dump(err_result, f)
        # the rest of the data may still be useful. save it in an HDF5.
        print("Saving all other data in HDF5...")
        output = np.array(output)
        # first remove err param values (err output already removed)
        param_values = param_values[[ii for ii in range(len(param_values)) if ii not in error_places],:]
        # convert to dataframe
        param_values = pd.DataFrame(param_values, columns=problem['names'])
        assert output.shape[0] == param_values.shape[0]
        # save as HDF5
        store = pd.HDFStore('nonerr_results.h5')
        store['param_values'] = param_values
        store['raw_output'] = pd.DataFrame(output, 
                            columns=observable_names)
        store.close()
        # cleanup temporary data dump and quit
        os.remove('raw_result_data.pickle')
        print("Please review output dump.")
        return

    ### No errors found! Save results in HDF5 as dataframe. ###
    print('Parsing the results...')
    output = np.array(output)
    param_values = pd.DataFrame(param_values, columns=problem['names'])
    store = pd.HDFStore(filename+'.h5')
    store['param_values'] = param_values
    store['raw_output'] = pd.DataFrame(output, 
                          columns=observable_names)
    # cleanup temporary data dump
    os.remove('raw_result_data.pickle')

    ########################################################################
    ###         Conduct Sobol analysis on the output data and save       ###
    ########################################################################

    ### Analyze the results for each observable and view using Pandas ###
    # For each observable, this occurs in three steps:
    #   1) analyze data (Sobol)
    #   2) pop out the second-order sensitivity analysis and confidence intervals
    #       to save them in a separate dict, S2
    #   3) save the rest of the analysis in a list

    sens = []
    S2 = {}
    for n, obs_name in enumerate(observable_names):
        # conduct analysis
        sens_result = sobol.analyze(problem, output[:,n], calc_second_order=True)
        # pop out the second-order sensitivity into dict as a DataFrame
        S2[obs_name] = pd.DataFrame(sens_result.pop('S2'), index=problem['names'],
                           columns=problem['names'])
        # pop out the second-order confidence interval into dict as DataFrame
        S2[obs_name+'_conf'] = pd.DataFrame(sens_result.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
        # append the rest of the data to the list as a DataFrame
        sens.append(pd.DataFrame(sens_result,index=problem['names']))


    ### Save the analysis in the HDF5 ###
    print('Saving...')
    for n, obs_name in enumerate(observable_names):
        store[obs_name+'_sens'] = sens[n]
    for key in S2.keys():
        store['S2/'+key] = S2[key]
    # Save the parameter bounds as a pandas Series (this was throwing a warning. ignore it.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store['bounds'] = pd.Series(problem['bounds'])
    # close the HDF5
    store.close()

    ### Plot and save ###
    plot_S1_ST_tbl(*sens, bounds=problem['bounds'], show=False)



def load_data(filename):
    '''Load analysis data from previous run and return (as a store object) 
    for examination.
    
    For use in ipython.'''

    return pd.HDFStore(filename)



def print_max_conf(store):
    '''Print off the max confidence interval for each variable in the store,
    for both first-order and total-order indices'''
    for var in ['S_sens', 'P_sens', 'A_sens', 'R_sens']:
        print('----------- '+var+' -----------')
        print('S1_conf_max: {}'.format(store[var]['S1_conf'].max()))
        print('ST_conf_max: {}'.format(store[var]['ST_conf'].max()))
        print(' ')



def plot_S1_ST_tbl_from_store(store, show=True, ext='pdf'):
    '''Produce sensitivity plot from a store object.  
    This is a wrapper around plot_S1_ST_tbl for convienence.'''
    S_sens = store['S_sens']
    P_sens = store['P_sens']
    A_sens = store['A_sens']
    R_sens = store['R_sens']
    bounds = list(store['bounds'])
    
    plot_S1_ST_tbl(S_sens, P_sens, A_sens, R_sens, bounds, show, ext)



def plot_S1_ST_tbl(S_sens, P_sens, A_sens, R_sens,
                   bounds=None, show=True, ext='pdf', startclr=None):
    '''Produces the sensitivity plots plus a table showing the range of the
    variables.
    
    The first arguments are the sensitivity data.
    bounds is a list of bounds on the input parameters'''

    # Gather the S1 and ST results
    all_names = ['S', 'P', 'A', 'R']
    all_results = [S_sens, P_sens, A_sens, R_sens]
    names = []
    results = []
    for n, result in enumerate(all_results):
        # Keep only the ones actually passed
        if result is not None:
            results.append(result)
            names.append(all_names[n])
    S1 = pd.concat([result['S1'] for result in results[::-1]], keys=names[::-1], axis=1) #produces copy
    ST = pd.concat([result['ST'] for result in results[::-1]], keys=names[::-1], axis=1)

    ##### Reorder (manual) #####
    # order = ['N', 'Rplus', 'v', 'beta', 'eta/alpha', 'theta/beta' ,'logDelta', 
    #          'gamma', 'delta', 'loglam']
    # bndorder = [0, 1, 2, -3, 5, 6, 4, -2, -1, 3]
    # S1 = S1.reindex(order)
    # ST = ST.reindex(order)
    # if bounds is not None:
    #     new_bounds = [bounds[ii] for ii in bndorder]
    #     bounds = new_bounds

    ###### Change to greek, LaTeX #####
    for id in S1.index:
        S1.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
    for id in ST.index:
        ST.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
    
    ###### Plot ######
    if bounds is not None:
        # setup for table
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,.35], wspace=.15, left=0.04,
                               right=0.975, bottom=0.15, top=0.915)
        axes = []
        for ii in range(2):
            axes.append(plt.subplot(gs[ii]))
    else:
        # setup without table
        fig, axes = plt.subplots(ncols=2, figsize=(13, 6))
    # Switch the last two colors so A is red
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    # colors = colors[:4]
    # clr = colors[-1]; colors[-1] = colors[-2]; colors[-2] = clr
    
    if startclr is not None:
        # Start at a different color
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[startclr:]
        s1bars = S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8, color=colors)
        s2bars = ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, color=colors, legend=False)
    else:
        s1bars = S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8)
        s2bars = ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, legend=False)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=18, rotation=-40) #-25
        ax.tick_params(axis='y', labelsize=14)
        #ax.get_yaxis().set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_title('First-order indices', fontsize=26)
    axes[1].set_title('Total-order indices', fontsize=26)
    handles, labels = s1bars.get_legend_handles_labels()
    s1bars.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=16)
    # Create table
    if bounds is not None:
        columns = ('Value Range',)
        rows = list(S1.index)
        # turn bounds into strings of ranges
        cell_text = []
        for bnd in bounds:
            low = str(bnd[0])
            high = str(bnd[1])
            # concatenate, remove leading zeros
            if low != "0" and low != "0.0":
                low = low.lstrip("0")
            if high != "0" and high != "0.0":
                high = high.lstrip("0")
            # raise any minus signs
            low = low.replace('-','\u00AF')
            high = high.replace('-','\u00AF')
            cell_text.append([low+"-"+high])
        tbl_ax = plt.subplot(gs[2])
        the_table = tbl_ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                    loc='center')
        the_table.set_fontsize(18)
        the_table.scale(1,2.3)
        the_table.auto_set_column_width(0)
        tbl_ax.axis('off')
    #plt.tight_layout()
    # reposition table
    pos = tbl_ax.get_position()
    newpos = [pos.x0 + 0.02, pos.y0, pos.width, pos.height]
    tbl_ax.set_position(newpos)
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.{}".format(time.strftime("%m_%d_%H%M"), ext))
    return (fig, axes)



if __name__ == "__main__":
    args = parser.parse_args()
    if args.ncores is None:
        with Pool() as pool:
            main(args.N, args.filename, args.ncores, pool)
    elif args.ncores > 1:
        with Pool(args.ncores) as pool:
            main(args.N, args.filename, args.ncores, pool)
    else:
        main(args.N, args.filename)
