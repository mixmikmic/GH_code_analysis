import numpy as  np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set('notebook')

from scipy.optimize import curve_fit



f(x, y, 1)

def f(x, y, deg):
    print(deg)
    p = np.polyfit(x, y, deg)   
    xp = np.array( [[xi**d for d in range(deg,-1,-1)]  for xi in x]).transpose()
    p = p.reshape(1,-1)
    print(xp,p)
    return (p@xp).ravel()

N=20
x = np.linspace(0,1,N)
y = x**2 +x/2 - 1 + np.random.normal(0,.03,N)

plt.figure(figsize=(10,10))
plt.plot(x,y,'.',markersize=10)
plt.plot(x,f(x,y,1))





sns.lmplot(x='',y='',data=fits)



