get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib.pyplot import *

plot([1,2,3,4])

plot([1,2,3,4],[1,4,9,16])

plot([1,2,3,4],[1,4,9,16],'or')  # 'o' for dots, 'r' for red

scatter([1,2,3,4],[1,4,9,16])

from numpy import *

x = linspace(-2,2)
y = x**3-x
plot(x,y)

x = linspace(-3,3)

fig = figure(figsize=figaspect(0.2))

ax = fig.add_subplot(131)
ax.plot(x,cos(x),color='b')
ax.set_title('Cosine')

ax = fig.add_subplot(132)
ax.plot(x,sin(x),color='r')
ax.set_title('Sine')

ax = fig.add_subplot(133)
ax.plot(x,cos(x),'b',x,sin(x),'r')
ax.set_title('Both')

fig, ax = subplots()
num = 1000
s = 121
x1 = linspace(-0.5,1,num) + (0.5 - random.rand(num))
y1 = linspace(-5,5,num) + (0.5 - random.rand(num))
x2 = linspace(-0.5,1,num) + (0.5 - random.rand(num))
y2 = linspace(5,-5,num) + (0.5 - random.rand(num))
x3 = linspace(-0.5,1,num) + (0.5 - random.rand(num))
y3 = (0.5 - random.rand(num))
ax.scatter(x1, y1, color='r', s=2*s, marker='^', alpha=.4)
ax.scatter(x2, y2, color='b', s=s/2, marker='o', alpha=.4)
ax.scatter(x3, y3, color='g', s=s/3, marker='s', alpha=.4)



