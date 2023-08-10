import pywt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cPickle as pkl
import pandas as pd
import cufflinks as cf
import sys
sys.path
sys.path.append('../..')
with open('../../../data/locs.tsv', 'r') as f_handle:
    chan_locs = pd.read_csv(f_handle, delimiter='\t').as_matrix()
    chan_locs = np.array(chan_locs[3:, 1:])
with open('../../../data/sub-0001_trial-01.pkl', 'rb') as f_handle:
    f_real = pkl.load(f_handle)
    f_real = f_real - np.mean(f_real, axis = 1).reshape(-1, 1)
import cPickle as pkl
import bench
import bench.dataset_creation as dc           
import bench.disc_set as d_set          
import bench.utilities as ut
import bench.discriminibility as disc
import methods.viz as viz
import methods.denoise as den
import methods.interpolation as inter
import os
wave = 'db2'
params = {'p_global': {'chan_locs': chan_locs,
                       'wave': wave,
                       'k': 2,
                       'loc_unit': 'radians',
                       'verbose': True},
          'p_local': {}
} 
C = [pywt.wavedec(f_real[c, :], wave) for c in range(f_real.shape[0])]
f_real = inter.wavelet_coefficient_interp((f_real, [0, 10, 20, 30, 40, 50]), params['p_local'], params['p_global'])
C_den = [pywt.wavedec(f_real[c, :], wave) for c in range(f_real.shape[0])]
for i in range(10):
    viz.cross_compare(C, C_den, i)

wave = 'db2'
params = {'p_global': {'chan_locs': chan_locs,
                       'wave': wave,
                       'k': 5,
                       'loc_unit': 'radians',
                       'verbose': True},
          'p_local': {}
} 
C = [pywt.wavedec(f_real[c, :], wave) for c in range(f_real.shape[0])]
f_real = inter.wavelet_coefficient_interp((f_real, [0, 10, 20, 30, 40, 50]), params['p_local'], params['p_global'])
C_den = [pywt.wavedec(f_real[c, :], wave) for c in range(f_real.shape[0])]
for i in range(10):
    viz.cross_compare(C, C_den, i)



