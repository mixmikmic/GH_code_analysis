# Setup the environment
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import nengo
from nengo.dists import Choice
from nengo.utils.functions import piecewise

model = nengo.Network(label='Nonlinear Function')
with model:
    #Input - Piecewise step functions
    inputX = nengo.Node(piecewise({0: -0.75, 1.25: 0.5, 2.5: -0.75, 3.75: 0}))
    inputY= nengo.Node(piecewise({0: 1, 1.25: 0.25, 2.5: -0.25, 3.75: 0.75}))
    
    #Five ensembles containing LIF neurons
    X = nengo.Ensemble(100, dimensions=1, radius=1)               # Represents inputX
    Y = nengo.Ensemble(100, dimensions=1, radius=1)               # Represents inputY
    vector2D = nengo.Ensemble(224, dimensions=2, radius=2)        # 2D combined ensemble
    result_square = nengo.Ensemble(100, dimensions=1, radius=1)   # Represents the square of X
    result_product = nengo.Ensemble(100, dimensions=1, radius=1)  # Represents the product of X and Y
    
    #Connecting the input nodes to the appropriate ensembles
    nengo.Connection(inputX, X)
    nengo.Connection(inputY, Y)
    
    #Connecting input ensembles A and B to the 2D combined ensemble
    nengo.Connection(X, vector2D[0])
    nengo.Connection(Y, vector2D[1])
    
    #Defining a function that computes the product of two inputs
    def product(x):
        return x[0] * x[1]
    
    #Defining the squaring function
    def square(x):
        return x[0] * x[0]
    
    #Connecting the 2D combined ensemble to the result ensemble 
    nengo.Connection(vector2D, result_product, function=product)
    
    #Connecting ensemble A to the result ensemble
    nengo.Connection(X, result_square, function=square)

#Import the nengo_gui visualizer to run and visualize the model.
from nengo_gui.ipython import IPythonViz
IPythonViz(model, "non_linear.py.cfg")

from IPython.display import Image
Image(filename='non_linear.png')

