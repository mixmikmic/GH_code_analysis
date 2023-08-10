# Import matplotlib (plotting) and numpy (numerical arrays).
# This enables their use in the Notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

# Import IPython's interact function which is used below to
# build the interactive widgets
from IPython.html.widgets import interact



def plot_chain_distribution(n_carb = 1):
    
    # a is the chain grow probabilitie 
    
    a = np.linspace(0,1,100)
    w = n_carb * (1-a)**2 * a **(n_carb -1) 

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Chain grow probability ')
    ax.set_ylabel('Weight fraction')
    ax.set_title('Anderson Schultz Flory model')
    
 

    ax.plot(a,  w,  marker='o', linewidth=2)

# Interact function creates a user interface to see how variables affect the system.

interact(plot_chain_distribution, n_carb = (1,50,1));



catalyst = ['Ni','Fe','Co','Co/K']

def plot_density_function(catalyst = 'Fe'):
    
    # a is the chain grow probability.
    # Each grow probability is related with a catalyst due its properties so here association its made.
    
    if catalyst == 'Ni':
        a = 0.05
    if catalyst == 'Fe':
        a = 0.675
    if catalyst == 'Co':
        a = 0.775
    if catalyst == 'Co/K':
        a = 0.9

    
    n_carb = np.linspace(1,20,100)
    w = n_carb * np.power((1-a),2) * np.power(a,(n_carb -1)) 

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Number of carbons ')
    ax.set_ylabel('Weight fraction')
    ax.set_title('Anderson Schultz Flory model - density function')
    
 

    ax.plot(n_carb,  w,  marker='o', linewidth=2)

# Interact function creates a user interface to see how variables affect the system.


interact(plot_density_function, catalyst = ['Ni','Fe','Co','Co/K']) ;


