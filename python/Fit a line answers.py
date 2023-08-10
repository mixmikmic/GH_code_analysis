get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x=np.array([1.,3,4,6])
y=np.array([2,3,4,5])

xy=x*y
n=len(x)
m=(n*sum(xy)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)**2)
b=(sum(x*x)*sum(y)-sum(xy)*sum(x))/(n*sum(x*x)-sum(x)**2)
x_fit=np.linspace(min(x),max(x))
y_fit=m*x_fit+b

fig,ax=plt.subplots()
ax.scatter(x,y)
ax.plot(x_fit,y_fit)
ax.set_xlabel('x')
ax.set_ylabel('y')
title='The best fit line as a slope m={:.3f} and intercept b={:.3f}'.format(m,b)
ax.set_title(title)

KNYC=np.array([90.,91,93,85,83,87,92])
KLGA=np.array([88,88,93,84,80,84,90])

x=KNYC
y=KLGA
xy=x*y
n=len(x)
m=(n*sum(xy)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)**2)
b=(sum(x*x)*sum(y)-sum(xy)*sum(x))/(n*sum(x*x)-sum(x)**2)
x_fit=np.linspace(min(x),max(x))
y_fit=m*x_fit+b

fig,ax=plt.subplots()
ax.scatter(x,y)
ax.plot(x_fit,y_fit)
ax.set_xlabel('KNYC')
ax.set_ylabel('KLGA')

linregress_out=stats.linregress(x,y)

title=('The best fit line as a slope m={:.3f} and intercept b={:.3f}'.format(m,b)+
       '\nbest fit line linregress slope m={:.3f} and intercept b={:.3f} '.format(linregress_out[0],linregress_out[01]))
ax.set_title(title)

KNYC=np.array([90.,91,93,85,83,87,92])
KLGA=np.array([88,88,93,84,80,84,90])

fig,ax=plt.subplots()
fig.set_size_inches(6,6)
ax.scatter(x,y)

ax.set_xlabel('KNYC ($^{\circ}$F)')
ax.set_ylabel('KLGA ($^{\circ}$F)')
results=stats.linregress(x,y)

x_fit=np.linspace(min(x),max(x))
y_fit=results[0]*x_fit+results[1]
ax.plot(x_fit,y_fit)

props=dict(boxstyle='round',facecolor='white',alpha=0.5)
textstr='m={:.3f}\nb={:.3f}\n$r^2$={:.3f}\np={:.3f}'.format(results[0],results[1],results[2]**2,results[3])
ax.text(0.05,0.95,textstr,transform=ax.transAxes,fontsize=10,verticalalignment='top',bbox=props)



