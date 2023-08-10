__depends__=['../results/sample_sim_data.pickle']
__dest__='../results/sample_sim_plot.pdf'

import numpy as np
import pickle
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='paper',font_scale=2.5)
get_ipython().magic('matplotlib inline')

with open(__depends__[0],'rb') as f:
    data = pickle.load(f)

mu,std = scipy.stats.norm.fit(data)

xmin,xmax = np.min(data),np.max(data)
x = np.linspace(xmin,xmax,1000)
gauss = scipy.stats.norm.pdf(x,mu,std)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.hist(data,alpha=0.75,normed=True)
ax.plot(x,gauss,linewidth=3,color=seaborn.color_palette('deep')[2])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\mathrm{PDF}$')
ax.set_xlim([xmin,xmax])
plt.savefig(__dest__,format='pdf',dpi=1000)
plt.show()



