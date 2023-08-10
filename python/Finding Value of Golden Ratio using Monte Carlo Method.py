import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import numpy as np

graph = []
min_val = 100

for i in range(1,10000000):
    x = np.random.rand()
    y = np.random.rand()
    val1 = (x+y)/(x*1.0)
    val2 = x/(y*1.0)
    diff = abs(val1 - val2)
    if(diff<min_val):
        min_val = diff
        graph.append(val2)
        gr = val2

print(gr)

plt.plot(graph)
plt.show()

plt.plot(graph)
plt.savefig("gr_converence.png")

plt.plot(graph[16:])
plt.savefig("gr_converence_smaller.png")



