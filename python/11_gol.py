import pysal as ps
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from scipy.stats import bernoulli

k = 8 # dimension of lattice

w = ps.lat2W(k,k,rook=False)

w.n

w.neighbors[0]

w.neighbors[45]

w.weights[0]

y = bernoulli.rvs(0.45,size=w.n)

y

wy = ps.lag_spatial(w,y)

wy

ywy = y*wy

lw23 = np.nonzero( (ywy==2) + (ywy==3) )

lw23

dw3 = (1-y) * wy

np.nonzero(dw3==3)

live_next = np.nonzero( (ywy==2) + (ywy==3) + (dw3==3) )

live_next

y[live_next]

y1 = np.zeros_like(y)

y1[live_next] = 1

y1

def generation(y,w):
    y1 = np.zeros_like(y)
    wy = ps.lag_spatial(w,y)
    ywy = y * wy
    live_next = np.nonzero( ( ywy == 2 ) + ( ywy == 3 ) + ( ( 1-y ) * wy == 3 ) )
    y1[live_next] = 1
    return y1
        

y = bernoulli.rvs(0.45,size=w.n)

y1 = generation(y,w)

y1

y

y2 = generation(y1,w)

y2

ngen=350
k = 50
w = ps.lat2W(k, k, rook=False)
#y = bernoulli.rvs(0.45,size=w.n)
y = np.zeros((w.n,))
#R-pentomino pattern
kd2 = k/2
top = kd2 +  k * ( kd2 - 1 )
topr = top + 1
midl = top + k -1
mid = midl + 1
bot = mid + k
y[[top, topr, midl, mid, bot]] = 1
results = {}
for i in xrange(ngen):
    y1 = generation(y,w)
    results[i] = y1
    if np.all(y == y1):
        break
    print i, y.sum(), y1.sum()
    y = y1
    

generations = np.zeros((ngen,))
living = np.zeros_like(generations)
keys = results.keys()
keys.sort()
for i in keys:
    generations[i] = i
    living[i] = results[i].sum()
    if not i%10:
        ymat = results[i]
        ymat.shape = (50,50)
        plt.imshow(ymat,cmap='Greys', interpolation='nearest')
        plt.title("Generation %d"%i)
        plt.show()
    

generations.shape

plt.plot(generations,living)

ymat = results[ngen-1]

ymat.shape

ymat.shape=(50,50)

plt.imshow(ymat, cmap='Greys', interpolation='nearest')
plt.title("Last Generation")

ymat = results[0]

ymat.shape = (50,50)

plt.imshow(ymat, cmap='Greys', interpolation='nearest')
plt.title('First Generation: R-pentomino')

ymat = results[ngen-2]
ymat.shape=(50,50)
plt.imshow(ymat, cmap='Greys', interpolation='nearest')
plt.title("Penultimate Generation")

ymat = results[ngen-1]
ymat.shape=(50,50)
plt.imshow(ymat, cmap='Greys', interpolation='nearest')
plt.title("Final Generation")



