__depends__=[]
__dest__="../results/sample_math_plot.pdf"

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='paper',font_scale=2.5)
get_ipython().magic('matplotlib inline')

orders = [0,1,2,3,4]
bessels = {}
x = np.linspace(0,20,100)
for o in orders:
    bessels['J'+str(o)] = scipy.special.jv(o,x)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
for key,o in zip(bessels,orders):
    ax.plot(x,bessels[key],label=r'$J_{%d}$'%(o),linewidth=3)
ax.set_ylabel(r'$J_{\alpha}(x)$')
ax.set_xlabel(r'$x$')
ax.set_title(r'Bessel functions of the first kind')
ax.legend(loc='best')
plt.savefig(__dest__,format='pdf',dpi=1000)
plt.show()



