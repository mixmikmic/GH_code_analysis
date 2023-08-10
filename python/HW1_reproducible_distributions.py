import os
import sys
import numpy as np
import pylab as pl
get_ipython().magic('pylab inline')

np.random.seed(100)

a = np.random.randn(100,2)

b = 2.5*np.random.randn(50,2)+5

print a

scatter1 = pl.scatter(a[:,0],a[:,1],color='blue')
scatter2 = pl.scatter(b[:,0],b[:,1],color='red')
#print np.mean(a)
#print np.mean(b)
#print np.std(a)
#print np.std(b)
pl.legend([scatter1,scatter2],
          ['mean='+"%.2f"%np.mean(a)+' std='+"%.2f"%np.std(a),
           'mean='+"%.2f"%np.mean(b)+' std='+"%.2f"%np.std(b)],loc=2)
                               

pl.show()

