__author__ = 'wbhimji'
import sys,os
#os.environ['THEANO_FLAGS']='device=gpu0'
sys.path.append('/home/wbhimji/atlas_dl/notebooks')
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from notebooks.plotting.plot_images import plot_example, plot_examples,  compile_saliency_function, show_images, plot_feature_maps,GuidedBackprop 
get_ipython().magic('matplotlib inline')
from notebooks.networks import binary_classifier as bc
from lasagne.nonlinearities import rectify as relu
from lasagne.layers import *
from notebooks.plotting.curve_plotter import plot_roc_curve
from notebooks.metrics.objectives import bg_rej_sig_eff, calc_ams, sig_eff_at

from notebooks.configs.setup_configs import setup_configs

#networks, fns = cb.build_network(configs, bc.build_layers(configs))

a = h5py.File('/home/wbhimji/delphes_combined_64imageNoPU/val.h5')

y = a['all_events']["y"][0:10000]
#EM = a['all_events']['histEM'][0:10000]
E = a['all_events']['hist'][0:10000]
weight  = a['all_events']['weight'][0:10000]
passSRarray = a['all_events']['passSR'][0:10000]

plot_example(E[y==1 ][3])

df = pd.DataFrame({'y': y, 'weight' : weight, 'passSR' : passSRarray})

np.unique(weight[y==0])

df.groupby(df['weight'])['passSR'].sum()/df.groupby(df['weight'])['passSR'].count()

keys=["hist", "weight", "normalized_weight", "y"]

d={k:a['all_events'][k][0:10] for k in keys}

pred = np.load("results/run169/pred.npy")

#df2 = pd.DataFrame({ 'pred' : pred})

SigCut = 0.97
BgCut = 0.00000000001
print "BgEfficiency: " + str(np.where(pred < BgCut)[0].shape[0] / float(pred.shape[0])*100) + "%"
print "SigEfficiency: " + str(np.where(pred > SigCut)[0].shape[0] / float(pred.shape[0])*100) + "%"
BgCutArray = pred < BgCut
SigCutArray = pred > SigCut

plot_roc_curve(pred, a['all_events']["y"][:],a['all_events']['weight'][:], a['all_events']['passSR'][:], 'val', './')

bg_rej_sig_eff( a['all_events']['passSR'][:],a['all_events']["y"][:],a['all_events']['weight'][:])

pred10k = pred[0:10000]

E[y == 0 & SigCutArray[0:10000]].shape

weight[(y == 1) & (passSRarray == 1) & (pred10k < 0.01) ]

weight[ (y == 0) & (passSRarray == 0) & (pred10k > 0.01) ]

plot_examples(E[(y == 0) & (passSRarray == 0) & (pred10k > 0.01)][0:16],4)

plot_examples(E[(y == 0) &  (passSRarray == 0) & SigCutArray[0:10000]][0:16],4)

configs = setup_configs()

networks, fns = configs["net"].build_network(configs, configs["net"].build_layers(configs))

saliency_fn = compile_saliency_function(networks['net'])

saliency, max_class = saliency_fn(np.expand_dims(np.expand_dims(E[(y == 1) & (passSRarray == 1) & SigCutArray[0:10000] ][1], axis=0),axis=0).astype("float32"))

show_images(E[(y == 1) & (passSRarray == 1) & SigCutArray[0:10000] ][1], saliency, max_class, "default gradient")

show_images(E[np.where(pred < BgCut)[0][2]], saliency, max_class, "default gradient")

a = [[1,2,4,5,6,7,8],[3,4,4,3,2,3,4],[5,6,6,4,2,2,4]]
np.lib.pad(a,((0,0),(0,2)),'wrap')

a = E[ (y == 1) & (passSRarray == 1) &  (pred < 0.01)[0:10000]][6]
b = np.lib.pad(a,((0,0),(0,1)),'wrap')

b.shape

plot_example(a)

plot_example(b)

plot_example(E[SigCutArray[0:10000] ][2])

plot_feature_maps(E[(y == 1) & (passSRarray == 1) & (pred > 0.995)[0:10000] ][2].astype("float32"), networks['net'], "./", name="best_bg")

y

passSR == 0 

y==0

E[(y == 1) & (passSRarray == 1) & ~ SigCutArray[0:100000] ]

BgCut = 0.000001
BgCutArray = pred < BgCut

E [ (y == 1) & (passSRarray == 1) &(pred < 0.01)[0:10000]].shape



