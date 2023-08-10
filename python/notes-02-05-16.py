import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
plt.plot(x,y)
plt.ylim([-2,2])
plt.xlim([0,2*np.pi])
plt.title('Graph of $y = \sin(x)$');

print(plt.style.available)

with plt.style.context('seaborn-pastel'):
    t = np.linspace(0,1,100)
    x = np.cos(t)
    y = np.sin(t)
    plt.plot(x,y)
    plt.title('Graph of $\cos(\pi t)$');

with plt.style.context('fivethirtyeight'):
    t = np.linspace(0,2*np.pi,1000)
    x = np.cos(t)
    y = np.sin(2*t)
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.plot(x,y)
    plt.title('Infinty Plot with 538 Style')

plt.style.use('ggplot')

t = np.linspace(0,2*np.pi,1000)
x = np.cos(t)
y = np.sin(2*t)
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.plot(x,y);

get_ipython().magic('pinfo plt.hist')

samples = np.random.randint(1,11,1000)
plt.hist(samples,bins=10,range=[1,11],align='left')
plt.xlim([0,11])
plt.title('Uniform Random Integers');

samples = np.random.randint(1,11,[2,1000]).sum(axis=0)
plt.hist(samples,bins=19,range=[1,21],align='left')
plt.xlim([0,21])
plt.title('Sum of 2 Uniform Random Variables');

