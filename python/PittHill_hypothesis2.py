# some programmatic housekeeping
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(215)
get_ipython().magic('matplotlib inline')

def ChooseN(moment):
    MAX_N = 1000000
    EPSILON = 1e-9
    prob_sum = 0
    for j in range(0, MAX_N):
        prob_sum = prob_sum + poisson.pmf(j, moment)
        if prob_sum >= 1-EPSILON:
            return j

def PoissonTriples_exact(moment):
    """
    This function computes the probability that triples of Poisson random variables
    contain their own rounded mean based on the formula given in Pitt & Hill, 2016.
    
    Parameters
    ----------
    moment : integer
              The mean-variance parameter of the Poisson distribution from which
              triples of Poisson random variables are to be generated.
    
    Returns
    -------
    prob : numeric
            The exact probability that triples of Poisson random variables contain
            their own rounded means.
    """
    N = ChooseN(moment)
    total = 0
    
    for j in list(range(2, N + 1)):
        for k in list(range(j, N + 1)):
            inner = poisson.pmf(k - np.floor(j / 2), moment) + ((j % 2) * poisson.pmf(k - np.floor(j / 2) - 1, moment))
            outer = poisson.pmf(k, moment) * poisson.pmf(k - j, moment)
            prob = outer * inner
            total = total + (6 * prob)
    return(total)

def ColumnNames():
    return ['col1', 'col2', 'col3', 'average']

def PreProcess(filepath, skiprows, usecols): 
    """
    This function reads data and add min, max, include_mean values.
    
    Parameters
    ----------
    filepath : filepath of the data
    skiprows: number of rows to skip from the csv file
    usecols: range of columns of data to read in.
               
    Returns
    -------
    data : The original count data and some added columns of new stats data.
    """
    print('Reading Data from \"{0}\"'.format(os.path.basename(filepath)))
    data = pd.read_csv(filepath, skiprows=skiprows,usecols=usecols,na_values=' ', header = None, names = ColumnNames() ).dropna(axis=0)
    data['col_min'] = data.apply(lambda row: min(row['col1'],row['col2'],row['col3']), axis=1)
    data['col_max'] = data.apply(lambda row: max(row['col1'],row['col2'],row['col3']), axis=1)
    data['col_median'] = data.apply(lambda row: np.median([row['col1'],row['col2'],row['col3']]), axis=1)
    data['col_gap'] = data['col_max']-data['col_min']
    data['complete'] = data['col_gap']>=2
    data['include_mean'] = data.apply(lambda row: ((row['col1'] == round(row['average']) or row['col2'] == round(row['average']) or 
                                                               row['col3'] == round(row['average'])) and row['complete']),axis=1)
    return(data)

data_dir = '../data/PittHill_OSFdata_2016/csv/'
rts_colony = PreProcess(os.path.join(data_dir,'Bishayee Colony Counts 10.27.97-3.8.01.csv'),3,range(3,7))

momentMeans = np.array(np.round(rts_colony['average'])) # means of each triple from the data
probs = np.zeros(len(momentMeans))

for i in range(len(momentMeans)):
    probs[i] = PoissonTriples_exact(momentMeans[i])

np.savetxt("../data/ppoibin_probs.txt", probs)

import subprocess
callr = ('rscript' + ' ' + '../src/02_PH_hypothesis2_ppoibin.R')
subprocess.call(callr, shell = True)

