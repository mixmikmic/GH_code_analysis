from sympy import *
init_printing()
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

f = Symbol('f')  # Function to approximate
f_approx = Symbol('fbar')   # Approximating function
w = Symbol('w') # weighting function
chi2 = Symbol('chi^2')

f, f_approx, w, chi2

M = Symbol('M', integer=True)
k = Symbol('k', integer=True,positive=True)
a = IndexedBase('a',(M,))  # coefficient
h = IndexedBase('h',(M,))  # basis function
ak = Symbol('a_k')  # Temporary symbols to make some derivatives easier
hk = Symbol('h_k')  #    Basis function (function of r)
hj = Symbol('h_j')
r = Symbol('r',positive=True)
j = Symbol('j',integer=True)
poly_approx = Sum(a[k]*h[k],(k,0,M))
poly_approx_j = Sum(a[j]*h[j],(j,0,M)) # replace summation variable
poly_approx

eq1 = Eq(chi2, Integral(w(r)*(f(r)-f_approx(r,ak))**2,r))
eq1

eq2 = Eq(0,diff(eq1.rhs, ak))
eq2

eq3 = Eq(diff(poly_approx,ak,evaluate=False), hk)
eq3

eq4 = Eq(0, Integral(eq2.rhs.args[0].subs(diff(f_approx(r,ak),ak),hk(r)),r))
eq4

eq5 = Eq(0, Integral(eq4.rhs.args[0].subs(f_approx(r,ak), poly_approx_j),r))
eq5

eq6 = Eq(0, Integral(-eq5.rhs.args[0]/2,r))
eq6

base7 = expand(eq6.rhs.args[0])
eq7 = Eq(Integral(-base7.args[1],r),Integral(base7.args[0],r))
eq7

int7 = eq7.lhs.args[0]
eq8 = Eq(Sum(a[j]*Integral(Mul(*int7.args[1:]),r),(j,0,M)), eq7.rhs)
eq8



