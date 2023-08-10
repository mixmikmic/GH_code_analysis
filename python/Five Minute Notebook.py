# you only need to do this once. Shamelessly stolen from Johannsen.
get_ipython().system('pip2 install --upgrade version_information')

#Preamble. These are some standard things I like to include in IPython Notebooks.
import astropy
from astropy.table import Table, Column, MaskedColumn
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('load_ext version_information')

get_ipython().magic('version_information numpy, scipy, matplotlib, sympy, version_information')

# special IPython command to prepare the notebook for matplotlib
#interactive plotting in separate window
#%matplotlib qt 
#interactive charts inside notebooks, matplotlib 1.4+
#%matplotlib notebook  
#normal charts inside notebooks
get_ipython().magic('matplotlib inline')

#This cell will download some gaia data file to your pwd
import urllib2
import gzip
some_zipped_gaia_file = urllib2.urlopen('http://cdn.gea.esac.esa.int/Gaia/gaia_source/csv/GaiaSource_000-010-207.csv.gz')
some_gaia_file_saved = open('GaiaSource_000-010-207.csv.gz','wb')
some_gaia_file_saved.write(some_zipped_gaia_file.read())
some_zipped_gaia_file.close()
some_gaia_file_saved.close()
some_gaia_zipfile = gzip.GzipFile('GaiaSource_000-010-207.csv.gz', 'r') 

from astropy.io import ascii
data = ascii.read(some_gaia_zipfile)

data

data['ra'].mean()

data['dec'].mean()

from numpy import random
random_subsample = data[random.choice(len(data), 10000)]

plt.scatter(random_subsample['ra'],random_subsample['dec'], s=0.1, color='black')

plt.xlabel('R.A.', fontsize=16)
plt.ylabel('Dec', fontsize=16)

