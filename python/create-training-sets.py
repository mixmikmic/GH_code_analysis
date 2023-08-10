get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from pymatgen import Composition
from itertools import product
import numpy as np
import pandas as pd
import os
import shutil

oqmd_data = pd.read_csv(os.path.join('..', 'oqmd_all.txt'), delim_whitespace=True)
print('Read %d entries'%len(oqmd_data))
oqmd_data.head()

for col in oqmd_data.columns:
    if col == 'comp': continue
    oqmd_data[col] = pd.to_numeric(oqmd_data[col], errors='coerce')

oqmd_data.query('delta_e > -20 and delta_e < 5', inplace=True)

oqmd_data['comp_obj'] = oqmd_data['comp'].apply(lambda x: Composition(x))

oqmd_data['pretty_comp'] = oqmd_data['comp_obj'].apply(lambda x: x.reduced_formula)

oqmd_data.sort_values('delta_e', ascending=True, inplace=True)
oqmd_data.drop_duplicates('pretty_comp', keep='first', inplace=True)
print('Reduced dataset to %d entries'%len(oqmd_data))

oqmd_data['nelems'] = oqmd_data['comp_obj'].apply(lambda x: len(x))

oqmd_data['system'] = oqmd_data['comp_obj'].apply(lambda x: "-".join([y.symbol for y in x]))

oqmd_data['system'].value_counts()[:10]

my_system = ["Na", "Fe", "Mn", "O"]

def get_test_data(elems):
    """Get the data that is in any of the phase diagrams that are subsets of a certain system
    
    Ex: For Na-Fe-O, these are Na-Fe-O, Na-Fe, Na-O, Fe-O, Na-Fe, Na, Fe, O
    
    :param elems: iterable of strs, phase diagram of interest
    :return: subset of OQMD in the constituent systems"""
    
    # Generate the constituent systems
    systems = set()
    for comb in product(*[elems,]*len(elems)):
        sys = "-".join(sorted(set(comb)))
        systems.add(sys)
    
    # Query for the data
    return oqmd_data.query(' or '.join('system=="%s"'%s for s in systems))

test_set = get_test_data(my_system)
print('Gathered a test set with %d entries'%len(test_set))
test_set.sample(5)

mad = np.abs(test_set['delta_e'] - test_set['delta_e'].mean()).mean()
std = test_set['delta_e'].std()
print('MAD: {:.3f} eV/atom'.format(mad))
print('Std Dev: {:.3f} eV/atom'.format(std))

train_set = oqmd_data.loc[oqmd_data.index.difference(test_set.index)]
print('Training set size is %d entries'%len(train_set))

def save_magpie(data, path):
    """Save a dataframe in a magpie-friendly format
    
    :param data: pd.DataFrame, data to be saved
    :param path: str, output path"""
    
    data[['comp','delta_e']].to_csv(path, index=False, sep=' ')

save_magpie(test_set, os.path.join('datasets', '%s_test_set.data'%(''.join(my_system))))

save_magpie(train_set, os.path.join('datasets', '%s_train_set.data'%(''.join(my_system))))

my_pair = ['Ti', 'O']

def get_test_data(elems):
    """Get the data that contains all of a certain set of elements.
        
    :param elems: iterable of strs, elems to exclude
    :return: subset of OQMD in the constituent systems"""
    
    # Process the dataset
    hit = oqmd_data['system'].apply(lambda x: all([e in x.split("-") for e in elems]))
    return oqmd_data[hit]

test_set = get_test_data(my_pair)
print('Gathered a test set with %d entries'%len(test_set))
test_set.sample(5)

mad = np.abs(test_set['delta_e'] - test_set['delta_e'].mean()).mean()
std = test_set['delta_e'].std()
print('MAD: {:.3f} eV/atom'.format(mad))
print('Std Dev: {:.3f} eV/atom'.format(std))

train_set = oqmd_data.loc[oqmd_data.index.difference(test_set.index)]
print('Training set size is %d entries'%len(train_set))

save_magpie(test_set, os.path.join('datasets', '%s_test_set.data'%('-'.join(my_pair))))

save_magpie(train_set, os.path.join('datasets', '%s_train_set.data'%('-'.join(my_pair))))



