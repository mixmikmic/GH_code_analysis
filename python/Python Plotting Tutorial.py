from numpy import *
from matplotlib.pyplot import *

get_ipython().magic('matplotlib inline')

x = linspace(-2*pi,2*pi,100) # 100 data points spaced linearly
y = cos(x)

plot(x,y,label="cos(x)")
xlabel("x")
ylabel("y")
title("hello world")
legend()
savefig('hello_world.pdf') # Save this as a pdf file

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats # We'll use this to make some random distributions for examples

from matplotlib import rcdefaults
rcdefaults()

get_ipython().magic('matplotlib inline')

x = np.linspace(-2*np.pi,2*np.pi,100)
y = np.cos(x)
plt.plot(x,y,label="cos(x)")

x = np.linspace(-2*np.pi,2*np.pi,500)
signal = np.cos(x)
noise = stats.norm().rvs(signal.shape) # Gets random deviates with a Gaussian distribution
plot(x,signal,label='cos(x)',linewidth=2) # Setting the linewidth as well as the label
scatter(x,signal+noise,label='random deviates',c=np.abs(noise),alpha=0.5) # Setting color and transparency
plt.colorbar(label='residual')
plt.ylim(-5,5)
plt.legend()

n,bins,patches = plt.hist(noise,normed=True,alpha=0.5) 
# n,bins,patches = plt.hist(noise,np.arange(-4,4.,0.1),alpha=0.5) 
# n,bins,patches = plt.hist(noise,np.arange(-4,4.,0.1),alpha=0.5,edgecolor='none') 
plt.plot(x,stats.norm().pdf(x))
plt.xlim(-4,4)

img = np.zeros((30,30),dtype=np.float64) # Set up an empty array
img[5,25] = 100
img[15,15] = 200
plt.imshow(img)
plt.colorbar()
plt.ylabel('row')
plt.xlabel('column')

plt.imshow(img.T,cmap=plt.cm.gray,vmin=-20,vmax=200.,interpolation='nearest',origin='lower')
plt.colorbar()
plt.xlabel('FITS first axis (x)')
plt.ylabel('FITS second axis (y)')
plt.title('ds9 orientation')

fig = plt.figure() 
ax = fig.add_subplot(1,1,1)
line, = ax.plot(x,signal,label='cos(x)') 
# Note the comma in the line above. 
# This is because ax.plot returns a tuple; in this case there is just one element 

help(line)

line.set_linewidth(2)
line.set_color('g')
fig

ax.set_ylabel(r"$\cos(\theta)$")
ax.set_xlabel(r"$\theta ({\rm radians})$")
ax.xaxis.label.set_fontsize(20)
ax.set_xlim(-6,6)
fig

ax2 = ax.twiny() # This function puts a second axis at the top
degrees = 180.*x/np.pi
topticks = [] # Set up an empty list for the top ticks
for t in ax.get_xticks():
    topticks += [180.*t/np.pi] # Convert to degrees
ax2.set_xticks(topticks)
ax2.set_xlabel(r'$\theta ({\rm degrees})$',fontsize=20)
fig

plt.subplot(2,1,1)
plt.plot(x,signal)
plt.subplot(2,1,2)
plt.plot(x,signal/(1+signal**2))

from matplotlib import style
style.available

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(x,signal,label='cos(x)')
ax.set_xlabel(r'$\theta ({\rm radians})$')

plt.style.use('classic')
plt.xkcd()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(x,signal,label='cos(x)')
ax.set_xlim(-1,5)
ax.set_ylim(-1.2,1.2)

ax.set_ylabel('level of attention')
ax.text(0.5,1.0,"Coffee")
ax.text(3.2,-0.5,"Look, a squirrel!",horizontalalignment='center')
fig

plt.setp(ax.get_xticklabels(),visible=False)
plt.setp(ax.get_xticklines(),visible=False)
fig

