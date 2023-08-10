#Setup for the notebook
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

import nengo
from nengo import spa
from nengo.spa import Vocabulary

#Number of dimensions for the Semantic Pointers
dimensions = 16

#Make a model object with the SPA network
model = spa.SPA(label='Routed_Sequence with cleanupA', seed=12)

with model:
    #Specifying the modules to be used
    model.state = spa.State(dimensions=dimensions, feedback=1, feedback_synapse=0.01)
    model.vision = spa.State(dimensions=dimensions) 
    
    # Specify the action mapping
    actions = spa.Actions(
        'dot(vision, START) --> state = vision',
        'dot(state, A) --> state = B',
        'dot(state, B) --> state = C',
        'dot(state, C) --> state = D',
        'dot(state, D) --> state = E',
        'dot(state, E) --> state = A'
    )
    
    #Creating the BG and Thalamus components that confirm to the specified rules
    model.BG = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.BG)
    
    #Change the seed of this RNG to change the vocabulary
    rng = np.random.RandomState(0)
    vocab = Vocabulary(dimensions=dimensions)

    #Create the transformation matrix (pd) and the cleanup ensemble (cleanupA) 
    pd = [vocab['A'].v.tolist()] 
    model.cleanup = spa.State(neurons_per_dimension=100, dimensions=1)
    
    #Function that provides the model with an initial input semantic pointer.
    def start(t):
        if t < 0.4:
            return '0.8*START+D'
        else:
            return '0'

    #Input
    model.input = spa.Input(vision=start)
    
    #Projecting the state of the cortex on to the cleanup ensemble using a transformation matrix 'pd'.
    nengo.Connection(model.state.output, model.cleanup.input, transform=pd)

#Import the nengo_gui visualizer to run and visualize the model.
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "spa_sequencerouted_cleanup.py.cfg")

from IPython.display import Image
Image(filename='spa_sequencerouted_cleanup.png')

