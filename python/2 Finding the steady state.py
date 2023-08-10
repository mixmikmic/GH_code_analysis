# Setting up a custom stylesheet in IJulia
from IPython.core.display import HTML
from IPython import utils  
import urllib2
HTML(urllib2.urlopen('http://bit.ly/1Bf5Hft').read())
#HTML("""
#<style>
#open('style.css','r').read()
#</style>
#""")

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
sym.init_printing() 

import pypwt
import solowpy

# define model parameters
ces_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.02, 'n': 0.03, 's': 0.15,
              'delta': 0.05, 'alpha': 0.33, 'sigma': 0.95}

# create an instance of the solow.Model class
ces_model = solowpy.CESModel(params=ces_params)

# check the docstring...
get_ipython().magic('pinfo ces_model.steady_state')

ces_model.steady_state

get_ipython().magic('pinfo solowpy.Model.find_steady_state')

k_star, result = ces_model.find_steady_state(1e-6, 1e6, method='bisect', full_output=True)

print("The steady-state value is {}".format(k_star))
print("Did the bisection algorithm coverge? {}".format(result.converged))

valid_methods = ['brenth', 'brentq', 'ridder', 'bisect']

for method in valid_methods:
    actual_ss = ces_model.find_steady_state(1e-6, 1e6, method=method)
    expected_ss = ces_model.steady_state

    print("Steady state value computed using {} is {}".format(method, actual_ss)) 
    print("Absolute error in is {}\n".format(abs(actual_ss - expected_ss)))

valid_methods = ['brenth', 'brentq', 'ridder', 'bisect']

for method in valid_methods:
    print("Profiling results using {}:".format(method)) 
    get_ipython().magic('timeit -n 1 -r 3 ces_model.find_steady_state(1e-6, 1e6, method=method)')
    print("")



