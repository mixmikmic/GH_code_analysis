import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# func is the function we are using: Remember we need x = f(x)
# k is the tolerance value
def fpi(x0,g,tol):
    x = list()
    x_old = x0
    x.append(x0)
    error = 1 
    while error > tol:
        x_new = g(x_old)
        #print(x_new)
        error = np.abs(x_new - x_old)
        #print(error)
        x.append(x_new)
        x_old = x_new
        if error > 100:
            break
    return pd.Series(x)

g1 = lambda x: (2*x+2)**(1/3)
g2 = lambda x: np.log(7-x)
g3 = lambda x: np.log(4-np.sin(x))
g4 = lambda x: (np.cos(x))**2

sol1 = fpi(0.5,g1,10**(-8))
print(sol1)

sol2 = fpi(0.5,g2,10**(-8))
print(sol2)

sol3 = fpi(0.5,g3,10**(-8))
print(sol3)

sol4 = fpi(0.5,g4,10**(-6))
print(sol4)



