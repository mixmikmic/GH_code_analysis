import numpy as np
import time
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

ti=time.clock()


def randpole (p, N):
    pole=np.random.rand(N,N)
    for i in range (0,N):
        for j in range (0,N):
            if pole[i,j] <=p: pole[i,j]=1
            else: pole[i,j]=0
    return pole

p= 0.5
N= 30

X = randpole(p,N)
Y = randpole(p,N)

nula = randpole(0,N)


plt.figure(figsize=(10,10))

plt.xlim(0,N)
plt.ylim(0,N)

plt.xticks([])
plt.yticks([])

plt.quiver(X, nula, scale = N, color='darkblue', headaxislength = 1)
plt.quiver(nula, Y, scale = N, color='darkblue', headaxislength = 1)

plt.show()

