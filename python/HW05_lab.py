######################################################################
## inverse CDF for exponential distribution
## with parameter lambda -- calculation by hand
######################################################################

def invCDF(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([])
    if( np.sum(arr<0) + np.sum(arr>=1) > 0 ):
        print('wrong input, should be in [0,1)')
        return np.array([])

    return -np.log(1-arr)/lam

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats

def invCDF(arr,lam):
    if( type(arr) != type(np.array([])) ):
        try:
            arr = np.array(arr,dtype=float)
        except:
            print('wrong input for x')
            return np.array([])
    if( np.sum(arr<0) + np.sum(arr>=1) > 0 ):
        print('wrong input, should be in [0,1)')
        return np.array([])

    return -np.log(1-arr)/lam

# This is an example: to generate 10 random numbers follows exp(1)
# X = generate 10 random numbers exp(1)

u = np.random.rand(10)
X = invCDF(u,1)
print(X)



