import sys
sys.path.append('../models/')

from ipywidgets import interact
from gr4j import interaction
get_ipython().magic('matplotlib inline')

river_name = 'Pur River'
path_to_scheme = '../data/pur_scheme.csv'
path_to_observations = '../data/pur_observations.csv'

interact(interaction, 
         river_name=river_name, 
         path_to_scheme=path_to_scheme, 
         path_to_observations=path_to_observations,
         X1=(0, 1500, 100), X2=(-10, 5, 0.5), X3=(0, 500, 10), X4=(0, 4, 0.1),
         __manual=True)



