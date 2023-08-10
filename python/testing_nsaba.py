get_ipython().magic('matplotlib inline')
from nsaba.nsaba import Nsaba
from nsaba.nsaba.visualizer import NsabaVisualizer
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools

get_ipython().magic('load_ext line_profiler')

# Simon Path IO

data_dir = '../../data_dir'
os.chdir(data_dir)
Nsaba.aba_load()
Nsaba.ns_load()

#Torben Path IO

ns_path = "/Users/Torben/Documents/ABI analysis/current_data_new/"
aba_path = '/Users/Torben/Documents/ABI analysis/normalized_microarray_donor9861/'

Nsaba.aba_load(aba_path)
Nsaba.ns_load(ns_path)

# Loading gene expression for all ABA registered Entrez IDs.
A = Nsaba()
A.load_ge_pickle('Nsaba_ABA_ge.pkl')

get_ipython().magic("time A.get_ns_act('attention', thresh=-1)")
A.get_ns_act('reward', thresh=-1)

# Testing ge_ratio()
A = Nsaba()
A.ge_ratio((1813,1816))



rand = lambda null: np.random.uniform(-10,10,3).tolist()
coord_num = 20
coords = [rand(0) for i in range(coord_num)]

A.coords_to_ge(coords, entrez_ids=[1813,1816], search_radii=8)

A.get_aba_ge([733,33,88])

A.get_ns_act("attention", thresh=-1, method='knn')

# You can use the sphere method too, if you want to weight by bucket. 
# e.g:
# A.get_ns_act("attention", thresh=.3, method='sphere')

A.make_ge_ns_mat('attention', [733, 33, 88])

A.make_ge_ns_mat('attention', [733, 33, 88])



NV = NsabaVisualizer(A)

NV.visualize_ge([1813])

NV.visualize_ns('attention', alpha=.3)

NV.lstsq_ns_ge('attention', [1813])

NV.lstsq_ge_ge(1813, 1816);

NV.lstsq_ns_ns('attention', 'reward')



