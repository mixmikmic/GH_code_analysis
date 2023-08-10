import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# This line tells the notebook that when we make a plot, we want to see it right away

x = np.array([  6.75696653,   2.63719654,   4.11100328,   5.92035153,
         8.33389304,  -0.22703768,   7.27983009,   8.04383391,
        10.70744198,   2.6992453 ])

x

x**2

2*x

np.sin(x)

y = np.cos(x)

y

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('cosine(x)')

x = np.arange(0,10,0.1)
print(x)
y = np.cos(x)
print(y)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('cosine(x)')

