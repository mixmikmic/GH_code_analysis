#Setup the environment
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import nengo
from nengo import spa      #import spa related packages

#Number of dimensions for the Semantic Pointers
dimensions = 16

#Make a model object with the SPA network
model = spa.SPA(label='Routed_Sequence', seed=20)

with model:
    #Specify the modules to be used
    model.state = spa.State(dimensions=dimensions, feedback=1, feedback_synapse=0.01)
    model.vision = spa.State(dimensions=dimensions) 
    
    #Specify the action mapping
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
    
    #Function that provides the model with an initial input semantic pointer.
    def start(t):
        if t < 0.4:
            return '0.8*START+D'
        else:
            return '0'

    #Input
    model.input = spa.Input(vision=start)

#Import the nengo_gui visualizer
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "spa_sequencerouted.py.cfg")

from IPython.display import Image
Image(filename='spa_sequencerouted.png')

