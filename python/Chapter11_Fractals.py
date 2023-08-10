# %load science.py
get_ipython().run_line_magic('matplotlib', 'notebook')
from sympy import init_session, symbols
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import theano as T
#from sympy.geometry import *
init_session()
a,b,c = symbols('a b c')
r, theta, phi = symbols('r theta phi', positive=True)
print('Loaded a b c theta and phi')
print('Load Theano as T')

#What is the limit of  a set? 
limit(Rational(2,3)**n, z=n, z0=oo, dir='+')

#Dimension of Self-similar Fractals.
n_copies = float(input("How many copies does one rescaling create? "))
r_scale_factor = float(input("What is the rescaling factor? "))
dimension = np.log(n_copies)/np.log(r_scale_factor)
print("The dimension, d, is {0:.2f}".format(dimension))

#out = np.dot(arr_one,arr_two.T) #MxT; NxT arrays
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

a1 = np.random.rand(10,3)
a2 = np.random.rand(10,3)
ans = corr2_coeff(a1,a2)
ans.shape



