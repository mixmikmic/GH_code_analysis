__depends__=[]
__dest__="../results/sample_exp_plot.pdf"

import pickle
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='paper',font_scale=2.5)
get_ipython().magic('matplotlib inline')

with open('../results/static/sample_exp_data.pickle','rb') as f:
    results = pickle.load(f)

def linear_fit(x,a,b):
    return a*x + b

pop,covar = scipy.optimize.curve_fit(linear_fit,results['x'],results['y'])

y_fit = linear_fit(results['x'],*pop)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.scatter(results['x'],results['y'],marker='.',s=200,label='data',color=seaborn.color_palette('deep')[0])
ax.plot(results['x'],y_fit,linewidth=3,color=seaborn.color_palette('deep')[2],label='fit')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend(loc='best')
plt.savefig(__dest__,format='pdf',dpi=1000)
plt.show()



