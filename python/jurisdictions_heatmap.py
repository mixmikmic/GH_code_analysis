import sys

sys.path.append('../../code/')
import os
import json
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

import networkx as nx

from load_data import load_citation_network, case_info
from helper_functions import *

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

data_dir = '../../data/'
court_name = 'scotus'


court_adj_mat = pd.read_csv(data_dir + 'clean/jurisdictions_adj_mat.csv', index_col='Unnamed: 0')
court_adj_mat.index = [j + '_ing' for j in court_adj_mat.index]
court_adj_mat.columns= [j + '_ed' for j in court_adj_mat.columns]

fed_appellate = ['ca' + str(i+1) for i in range(11)]
fed_appellate.append('cafc')
fed_appellate.append('cadc')

fed_appellate_ing = [j + '_ing' for j in fed_appellate]
fed_appellate_ed = [j + '_ed' for j in fed_appellate]

fed_appellate_network = court_adj_mat.loc[fed_appellate_ing, fed_appellate_ed]

fed_appellate_network

import seaborn.apionly as sns

Gn = fed_appellate_network.apply(lambda c: c/sum(c), axis=1)

plt.figure(figsize=[15, 15])
sns.heatmap(Gn,
            square=True,
            xticklabels=5,
            yticklabels=5);



