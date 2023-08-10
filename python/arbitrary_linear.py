#Setup the environment
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import nengo

#Create a 'model' object to which we can add ensembles, connections, etc.  
model = nengo.Network(label="Arbitrary Linear Transformation")
with model:
    #Two-dimensional input signal with constant values of 0.5 and -0.5 in two dimensions
    input = nengo.Node(lambda t: [.5,-.5])
      
    #Ensembles with 200 LIF neurons having dimentions 2 and 3
    x = nengo.Ensemble(200, dimensions=2)
    z = nengo.Ensemble(200, dimensions=3)
       
    #Connect the input to ensemble x
    nengo.Connection(input, x)
    
    #Connect ensemble x to ensemble z using a weight matrix
    weight_matrix = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]   
    nengo.Connection(x, z, transform = weight_matrix)

#Import the nengo_gui visualizer to run and visualize the model.
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "arbitrary_linear.py.cfg")

from IPython.display import Image
Image(filename='arbitrary_linear.png')

