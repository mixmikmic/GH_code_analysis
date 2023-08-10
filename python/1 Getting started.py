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

import numpy as np
import sympy as sym
import solowpy
sym.init_printing() 

get_ipython().magic('pinfo solowpy.Model.output')

# define model variables
A, K, L = sym.symbols('A, K, L')

# define production parameters
alpha, sigma = sym.symbols('alpha, sigma')

# define a production function
cobb_douglas_output = K**alpha * (A * L)**(1 - alpha)

rho = (sigma - 1) / sigma
ces_output = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)

get_ipython().magic('pinfo solowpy.Model.params')

# these parameters look fishy...why?
default_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.0, 'n': -0.03, 's': 0.15,
                  'delta': 0.01, 'alpha': 0.33}

# ...raises an AttributeError
model = solowpy.Model(output=cobb_douglas_output, params=default_params)

cobb_douglas_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.02, 'n': 0.03, 's': 0.15,
                       'delta': 0.05, 'alpha': 0.33}

cobb_douglas_model = solowpy.Model(output=cobb_douglas_output,
                                 params=cobb_douglas_params)

ces_params = {'A0': 1.0, 'L0': 1.0, 'g': 0.02, 'n': 0.03, 's': 0.15,
              'delta': 0.05, 'alpha': 0.33, 'sigma': 0.95}

ces_model = solowpy.Model(output=ces_output, params=ces_params)

get_ipython().magic('pinfo solowpy.Model.intensive_output')

ces_model.intensive_output

ces_model.evaluate_intensive_output(np.linspace(1.0, 10.0, 25))

get_ipython().magic('pinfo solowpy.Model.marginal_product_capital')

sym.simplify(ces_model.marginal_product_capital) 

ces_model.evaluate_mpk(np.linspace(1.0, 10.0, 25))

get_ipython().magic('pinfo solowpy.Model.k_dot')

ces_model.k_dot

ces_model.evaluate_k_dot(np.linspace(1.0, 10.0, 25))

get_ipython().magic('pinfo solowpy.cobb_douglas')

cobb_douglas_model = solowpy.CobbDouglasModel(params=cobb_douglas_params)

get_ipython().magic('pinfo solowpy.ces')

ces_model = solowpy.CESModel(params=ces_params)



