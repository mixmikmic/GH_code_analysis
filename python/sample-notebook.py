get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,1,0.03)
y1 = [4*i*(1-i) for i in x]
y2 = [1-i for i in y1]

plt.plot(x, y1,'.b')
plt.plot(x, y2,'*r')

plt.title('My Plot')
plt.xlabel("My x axis")
plt.ylabel("My y axis")
#plt.yscale('log')
#plt.xscale('log')

plt.show()

def fib1(n):
    if n<=2: 
        return 1
    else: 
        return fib1(n-1)+fib1(n-2)
    
def fib2(n):
    cur = 1
    prev = 0
    for _ in range(n-1):
        cur, prev = cur+prev, cur
    return cur

get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.yscale('log')
#plt.title('Fibbonaci execution time')
#plt.xlabel("N")
#plt.ylabel("Time to compute Fib(N)")
    
#plt.xscale('log')
import math

fig.show()

x = range(20,40)
y = [0]*len(x)
y2 = y[:]
for i in range(len(x)):
    result = get_ipython().run_line_magic('timeit', '-r1 -o -q fib1(i)')
    y[i]=math.log(result.best)
    y2[i] = y[0]+i*math.log((1+math.sqrt(5))/2)

    ax.clear()    
    ax.plot(x, y,'.b')
    ax.plot(x,y2,'*r')
    fig.canvas.draw()    


get_ipython().run_line_magic('timeit', 'fib1(20)')



