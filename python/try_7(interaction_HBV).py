import sys
sys.path.append('../models/')

from ipywidgets import interact
from hbv import interaction
get_ipython().magic('matplotlib inline')

river_name = 'Pur River'
path_to_scheme = '../data/pur_scheme.csv'
path_to_observations = '../data/pur_observations.csv'

# interact with new PE module
interact(interaction, 
         river_name=river_name, 
         path_to_scheme=path_to_scheme, 
         path_to_observations=path_to_observations,
         parBETA=(1, 6, 1), parCET=(0, 0.3, 0.1), parFC=(50, 500, 50), parK0=(0.01, 0.4, 0.05), 
         parK1=(0.01, 0.4, 0.05), parK2=(0.001, 0.15, 0.005), parLP=(0.3, 1, 0.1), parMAXBAS=(1, 7, 1), 
         parPERC=(0, 3, 0.2), parUZL=(0, 500, 50), parPCORR=(0.5, 2, 0.1), parTT=(-1.5, 2.5, 0.1), 
         parCFMAX=(1, 10, 1), parSFCF=(0.4, 1, 0.1), parCFR=(0, 0.1, 0.01), parCWH=(0, 0.2, 0.01), 
         __manual=True)



