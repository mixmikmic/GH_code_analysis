import sys
sys.path.append('../models/')

from ipywidgets import interact
from simhyd import interaction
get_ipython().magic('matplotlib inline')

river_name = 'Pur River'
path_to_scheme = '../data/pur_scheme.csv'
path_to_observations = '../data/pur_observations.csv'

interact(interaction, 
         river_name=river_name, 
         path_to_scheme=path_to_scheme, 
         path_to_observations=path_to_observations,
         INSC=(0,50,5), COEFF=(0,400,10), SQ=(0,10,1), SMSC=(1,1000,50), SUB=(0,1,0.1), 
         CRAK=(0,1,0.1), K=(0,1,0.1), etmul=(0.9,1.1,0.1), DELAY=(0.1,5,0.1), X_m=(0.01, 0.5, 0.01),
         __manual=True)



