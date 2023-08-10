import matplotlib as mpl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plot

from scipy import cos
from scipy import sin
from mpl_toolkits.mplot3d import Axes3D

#func = lambda x, y: (-cos(x))*(cos(y))*np.exp(-((x-np.pi)**2+(y-np.pi)**2))
func = lambda x, y: x**2+y**2

mpl.rcParams['legend.fontsize'] = 16
plot.xkcd()

x = np.arange(-100.,100.,0.05)
y = np.arange(-100.,100.,0.05)
z = func(x,y)

figure = plot.figure()
ax = figure.gca(projection='3d')

ax.plot_wireframe(x,y,z, label = 'Easom function')
ax.legend()

plot.show()


#example from mpl website
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

get_ipython().magic('matplotlib inline')
#example from mpl website
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 14

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plot

figure = plot.figure() # gets the plot of the graph
axes = figure.gca(projection='3d') # sets plot to 3 dimensions

#func = lambda x, y: (-cos(x))*(cos(y))*np.exp(-((x-np.pi)**2+(y-np.pi)**2))
func = lambda x, y: x**2+y**2

mpl.rcParams['legend.fontsize'] = 16
plot.xkcd()

x_domain = np.arange(-100.,100.,1.)
y_ = np.arange(-100.,100.,1.)
z = np.zeros(0)

for num in x:
    
    x = np.append(x,x)
    y = np.append(y,y)


axes.plot(x,y,z,"ro")

plot.show()

get_ipython().magic('matplotlib notebook')
#example from mpl website
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

figure = plt.figure()
axes = figure.gca(projection='3d') # sets plot to 3 dimensions

interval = 1.

x_domain = np.arange(-100.,151.,interval)
y_domain = np.arange(-100.,101.,interval)

x = np.zeros(0)
y = np.zeros(0)

for y_val in y_domain:
    
    x = np.append(x,x_domain)
    
    for x_val in x_domain:
       
        y = np.append(y,y_val)
        
func = lambda x, y: x**2+y**2

axes.plot(x,y,func(x,y),"p")

plt.show()

get_ipython().magic('matplotlib notebook')
#example from mpl website
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import cos

figure = plt.figure()
axes = figure.gca(projection='3d') # sets plot to 3 dimensions

interval = 0.8

x_domain = np.arange(-100.,151.,interval)
y_domain = np.arange(-100.,101.,interval)

x = np.zeros(0)
y = np.zeros(0)

for y_val in y_domain:
    
    x = np.append(x,x_domain)
    
    for x_val in x_domain:
       
        y = np.append(y,y_val)
        
func = lambda x, y: (-cos(x))*(cos(y))*np.exp(-((x-np.pi)**2+(y-np.pi)**2))

axes.plot(x,y,func(x,y),"r-") # plot the graph


plt.show()

