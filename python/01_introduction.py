import sys
sys.float_info

N = 100

# the exact answer for N = 100
f100 = 14466636279520351160221518043104131447711 / 2788815009188499086581352357412492142272

# initialize the sums
s1 = 0.0
s2 = 0.0

# forward sum
for n in range(1,N+1):
    s1 += 1.0/n

# backward sum
for n in range(N,0,-1):
    s2 += 1.0/n

# output results
print('f(100)  = %.16f'%f100)
print('s1      = %.16f'%s1)
print('s2      = %.16f'%s2)
print('|s1-s2| = %.2E' % (abs(s1-s2)))

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

N = 500
M = 50
mbset = np.zeros([N,N])

for j,x in enumerate(np.linspace(-2,1,N)):
    for i,y in enumerate(np.linspace(-1.0j,1.0j,N)):
        z0 = x + y
        z = 0
        for m in range(M):
            if abs(z) > 2:
                break
            z = z*z + z0
        mbset[i,j] = 1.0/m
        
plt.imshow(mbset,cmap='spectral',extent=[-2,1,-1,1])
plt.colorbar()
plt.xlabel(r'Re $z_0$')
plt.ylabel(r'Im $z_0$')



