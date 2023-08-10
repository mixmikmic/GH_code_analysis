get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


data=np.genfromtxt('GDP-Lifespan.csv',delimiter=',',skip_header=1,usecols=(2,3))
data=np.log10(data)
fit_order=1

GDP_fit=np.linspace(np.min(data[:,1]),np.max(data[:,1]))  #I set the range using the data
a=np.polyfit(data[:,1],data[:,0],fit_order)  #I just copied from above to have in the same cell
polynominal=np.poly1d(a)
lifespan_fit=polynominal(GDP_fit)

fig,ax=plt.subplots()
fig.set_size_inches(6,6)
ax.plot(GDP_fit,lifespan_fit)
ax.scatter(data[:,1],data[:,0])
ax.set_ylabel('log10 lifespan')
ax.set_xlabel('log10 GDP')
title='GDP vs lifespan with a polynominal with equation of\n '+str(np.poly1d(a))
ax.set_title(title)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as plt
from scipy import stats

data=np.genfromtxt('GDP-Lifespan.csv',delimiter=',',skip_header=1,usecols=(2,3))
data=np.log10(data)

results=stats.linregress(data[:,1],data[:,0])
GDP_fit=np.linspace(np.min(data[:,1]),np.max(data[:,1]))  #I set the range using the data
lifespan_fit=results[0]*GDP_fit+results[1]

fig,ax=plt.subplots()
fig.set_size_inches(6,6)
ax.plot(GDP_fit,lifespan_fit)
ax.scatter(data[:,1],data[:,0])
ax.set_ylabel('log10 lifespan')
ax.set_xlabel('log10 GDP')

props=dict(boxstyle='round',facecolor='white',alpha=0.5)
textstr='m={:.3f}\nb={:.3f}\n$r^2$={:.3f}\np={:.3f}'.format(results[0],results[1],results[2]**2,results[3])
ax.text(0.05,0.95,textstr,transform=ax.transAxes,fontsize=10,verticalalignment='top',bbox=props)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

fit_order=3

data=np.genfromtxt('GDP-Lifespan.csv',delimiter=',',skip_header=1,usecols=(2,3))


GDP_fit=np.linspace(np.min(data[:,1]),np.max(data[:,1]))  #I set the range using the data
a=np.polyfit(data[:,1],data[:,0],fit_order)  #I just copied from above to have in the same cell

polynominal=np.poly1d(a)
lifespan_fit=polynominal(GDP_fit)

fig,ax=plt.subplots()
fig.set_size_inches(6,6)
ax.plot(GDP_fit,lifespan_fit)
ax.scatter(data[:,1],data[:,0])
ax.set_xlabel('GDP')
ax.set_ylabel('Lifespan')
ax.set_xlim([0,60000])

props=dict(boxstyle='round',facecolor='white',alpha=0.5)
ax.text(0.05,0.95,polynominal,transform=ax.transAxes,fontsize=10,verticalalignment='top',bbox=props)

fig,ax=plt.subplots()
fig.set_size_inches(6,6)
data=np.genfromtxt('GDP-Lifespan.csv',delimiter=',',skip_header=1,usecols=(2,3))
ax.loglog((data[:,1]),(data[:,0]),linestyle='none',marker='o',basex=10,basey=10)
ax.grid(True,which="both",ls="-", color='0.65')
ax.set_xlabel('GDP')
ax.set_ylabel('Lifespan')



