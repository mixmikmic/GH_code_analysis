#To execute the Python code below, 
#click on this cell and type Shift-Enter on your keyboard.
#The workshop schedule should be displayed. 
from IPython.display import HTML
HTML('<iframe src=https://www.cta-observatory.org/indico/conferenceTimeTable.py?confId=1169#20161010 width=800 height=500></iframe>')

from IPython.display import HTML
HTML('<iframe src=http://www.psychopy.org width=600 height=300></iframe>')

from IPython.display import HTML
HTML('<iframe src=https://jakevdp.github.io/blog/2012/09/20/why-python-is-the-last/ width=600 height=300></iframe>')

# Type shift-enter to execute the code below

get_ipython().magic('pylab inline')

N = 200
r = 2 * rand(N)
theta = 2 * pi * rand(N)
area = 200 * r ** 2 * rand(N)
ax = plt.subplot(111, polar=True)
scatter(theta, r, c=theta, s=area, alpha=0.75)
show()

#Example:
2*2

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#plt.style.use('seaborn-talk') # different plotting styles. Useful to have large readable fonts
x=np.linspace(0,100)
plt.plot(x,x**2,'b.')

from ipywidgets import interact
def plotx(a,alpha,b,title='',ylog=False):
    x=np.linspace(0,100,num=1e3)
    y=a*x**alpha+b
    plt.plot(x,y)
    if ylog: plt.yscale('log')
    plt.title(title)
    plt.show()

interact(plotx,a=(1,10),alpha=(0.5,2,0.25),b=(0,5))    



