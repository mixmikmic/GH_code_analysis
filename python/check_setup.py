# -- If needed: Install some packages right now.

get_ipython().system('pip install lalsuite')
get_ipython().system('pip install gwpy')
get_ipython().system('pip install healpy')
get_ipython().system('pip install pycbc')

import numpy
import matplotlib
import lal
import lal.utils
import gwpy
import astropy
import pycbc

print("The software is ready!")

