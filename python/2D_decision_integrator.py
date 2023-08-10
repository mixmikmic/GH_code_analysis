#Setup the environment
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

import nengo
from nengo.processes import WhiteNoise
from nengo.utils.functions import piecewise
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform

model = nengo.Network(label='2D Decision Integrator', seed=11)
with model:
    #Input
    input1 = nengo.Node(-0.5)
    input2 = nengo.Node(0.5)
    
    #Ensembles 
    input = nengo.Ensemble(100, dimensions=2)
    MT = nengo.Ensemble(100, dimensions=2, noise=WhiteNoise(dist=Uniform(-0.3,0.3)))
    LIP = nengo.Ensemble(200, dimensions=2, noise=WhiteNoise(dist=Uniform(-0.3,0.3)))
    output = nengo.Ensemble(100, dimensions=2, noise=WhiteNoise(dist=Uniform(-0.3,0.3)))
     
    weight = 0.1
    #Connecting the input signal to the input ensemble
    nengo.Connection(input1, input[0], synapse=0.01)   
    nengo.Connection(input2, input[1], synapse=0.01) 
    
    #Providing input to MT ensemble
    nengo.Connection(input, MT, synapse=0.01) 
    
    #Connecting MT ensemble to LIP ensemble
    nengo.Connection(MT, LIP, transform=weight, synapse=0.1) 
    
    #Connecting LIP ensemble to itself
    nengo.Connection(LIP, LIP, synapse=0.1) 
    
    #Connecting LIP population to output
    nengo.Connection(LIP, output, synapse=0.01) 

#Import the nengo_gui visualizer to run and visualize the model.
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "2D_decision_integrator.py.cfg")

from IPython.display import Image
Image(filename='2D_decision_integrator.png')

