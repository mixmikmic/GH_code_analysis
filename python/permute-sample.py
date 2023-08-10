get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import division
import math
import numpy as np
import scipy as sp
from scipy.misc import comb, factorial
from scipy.optimize import brentq
from scipy.stats import chisquare, norm
import scipy.integrate
from random import Random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PIKK(n,k, gen=np.random):
    try: 
        x = gen.random(n)
    except TypeError:
        x = np.array([gen.random() for i in np.arange(n)])
    return set(np.argsort(x)[0:k])

def fykd(a, gen=np.random):
    for i in range(len(a)-2):
        J = gen.randint(i,len(a))
        a[i], a[J] = a[J], a[i]
    return(a)

fykd(np.arange(10))

def Random_Sample(n, k, gen=np.random):  # from Cormen et al.
    if k==0:
        return set()
    else:
        S = Random_Sample(n-1, k-1)
        i = gen.randint(1,n+1) 
        if i in S:
            S = S.union([n])
        else:
            S = S.union([i])
    return S

Random_Sample(100,5)

def R(n,k):  # Waterman's algorithm R
    S = np.arange(1, k+1)  # fill the reservoir
    for t in range(k+1,n+1):
        i = np.random.randint(1,t+1)
        if i <= k:
            S[i-1] = t
    return set(S)

R(100,5)

