# Setup the environment
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np 

import nengo
from nengo import spa                #import spa related packages
from nengo.spa import Vocabulary

#Number of dimensions for the Semantic Pointers
dimensions=16

#Make a model object with the SPA network
model = spa.SPA(label='Routed_Sequence with cleanupAll', seed=7)

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
    
    #Changing the seed of this RNG to change the vocabulary
    rng = np.random.RandomState(7)
    vocab = Vocabulary(dimensions=dimensions, rng=rng)
    
    #Creating the transformation matrix (pd_new) and cleanup ensemble (cleanup)  
    vsize = len((model.get_output_vocab('state').keys))

    pd = []
    for item in range(vsize):
        pd.append(model.get_output_vocab('state').keys[item])  
          
    pd_new = []
    for element in range(vsize):
        pd_new.append([vocab[pd[element]].v.tolist()]) 
        
    #cleanup = nengo.Ensemble(300, dimensions=vsize)
    model.cleanup = spa.State(neurons_per_dimension=300/vsize, dimensions=vsize)
    
    #Function that provides the model with an initial input semantic pointer.
    def start(t):
        if t < 0.4:
            return '0.8*START+D'
        else:
            return '0'

    #Input
    model.input = spa.Input(vision=start)
    
    #Projecting the state on to the cleanup ensemble using a transformation matrix 'pd'.
    for i in range(5):
        nengo.Connection(model.state.output, model.cleanup.input[i], transform=pd_new[i])  

#Import the nengo_gui visualizer to run and visualize the model.
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "spa_sequencerouted_cleanupAll.py.cfg")

from IPython.display import Image
Image(filename='spa_sequencerouted_cleanupAll.png')

