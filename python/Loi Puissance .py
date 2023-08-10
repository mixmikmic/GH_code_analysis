### importation
from random import randint
from random import uniform, seed
import sys
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import os
import seaborn

get_ipython().magic('pylab inline')

folder = 'seize_avril'
os.chdir(folder)

graph=json.load(open("undirected_graph_{}.json".format(folder))) ## importation of the data

len(graph) ### number of nodes in the new graph

h = [len(graph[key]) for key in graph.keys() if len(graph[key]) > 5] ## we keep the nodes which have at least 5 neighbors
h_2= [len(graph[key]) for key in graph.keys()] ## we keep everynodes

plt.hist(h_2,bins=50,range=(1,400))
plt.title("Power law for our Twitter Graph")
plt.show()

set_h = set(h)
x = [np.log(i) for i in set_h]
y = [np.log(h.count(i)/len(set_h)) for i in set_h]

plt.plot(x[0:150], y[0:150], 'ro')
plt.axis([1.5, 5.5, -10, 6])
plt.title(" Log- Log représentation")
plt.show()

x=[]
y=[]
for i in set(h):
    x+=[i]
    y+=[np.log(h.count(i))]

len(y)

plt.plot(x[0:150], y[0:150], 'ro')
plt.axis([0, 160, 0, 6])
plt.title("log(y)=f(x)")
plt.show()



