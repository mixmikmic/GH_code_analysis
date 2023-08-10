import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def newton_method(x0,f,df, tol):
    x = list()
    x_old = x0
    x.append(x0)
    error = 1
    while error > tol:
        x_new = x_old - f(x_old)/df(x_old)
        error = np.abs(x_new-x_old)
        x.append(x_new)
        x_old = x_new
        if error > 100:
            break
    return pd.Series(x)

x = Symbol('x')
y1 = Symbol('y1')
y2 = Symbol('y2')
y3 = Symbol('y3')
y4 = Symbol('y4')

y1 = x**3 - 2*x - 2
print(y1)
y2 = exp(x) + x - 7
print(y2)
y3 = x**5 + x - 1
print(y3)
y4 = log(x) + x**2 - 3
print(y4)

dy1 = diff(y1,x)
print(dy1)
dy2 = diff(y2,x)
print(dy2)
dy3 = diff(y3,x)
print(dy3)
dy4 = diff(y4,x)
print(dy4)

f1 = lambda x: x**3 - 2*x - 2
f2 = lambda x: np.exp(x) + x - 7
f3 = lambda x: x**5 + x - 1
f4 = lambda x: np.log(x) + x**2 - 3

df1 = lambda x: 3*(x**2) - 2
df2 = lambda x: np.exp(x) + 1
df3 = lambda x: 5*(x**4) + 1
df4 = lambda x: 2*x + 1/x

sol1 = newton_method(1,f1,df1,10**(-8))
print(sol1)

sol2 = newton_method(1,f2,df2,10**(-8))
print(sol2)

sol3 = newton_method(1,f3,df3,10**(-8))
print(sol3)

sol4 = newton_method(1,f4,df4,10**(-8))
print(sol4)

y5 = Symbol('y5')
y5 = 14*x*exp(x-2) - 12*exp(x-2)- 7*x**3 + 20*x**2 - 26*x + 12
print(y5)

dy5 = diff(y5,x)
print(dy5)

f5 = lambda x: 14*x*np.exp(x-2) - 12*np.exp(x-2)- 7*x**3 + 20*x**2 - 26*x + 12
df5 = lambda x: -21*x**2 + 14*x*np.exp(x - 2) + 40*x + 2*np.exp(x - 2) - 26

z = np.linspace(0,3,100)
plt.plot(z,f5(z),'r')
plt.axhline(y=0)
plt.axvline(x=0)

sol5 = newton_method(0.5,f5,df5,10**(-8))
print(sol5)

sol6 = newton_method(1.5,f5,df5,10**(-6))
print(sol6)



