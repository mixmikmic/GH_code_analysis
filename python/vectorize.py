
import numpy as np
from random import randint

pya = [randint(0,100) for _ in range(100)]
pyb = [randint(0,100) for _ in range(100)]

npa = np.array(pya)
npb = np.array(pyb)




# traditional
for i, _ in enumerate(pya):
    pya[i] += 3

# numpy
npa += 3

assert(np.all(npa == pya))

# even more significant for deeper structures

# copy rows?

pysq = []
for i in range(len(pya)):
    pysq.append(pya[:])
    
w=npa.shape[0]
npsq = np.tile(npa, w).reshape((w,w))


def pyargmaxsq(sq):
    mv=float('-inf')
    mi=-1
    mj=-1
    for i, row in enumerate(sq):
        for j, v in enumerate(row): 
            if v > mv:
                mv=v
                mi=i
                mj=j
    return (mv, (mi, mj))
        
def npargmaxsq(sq):
    mv = sq.max()
    i = sq.argmax()
    mi, mj = np.unravel_index(i, sq.shape)
    return (mv, (mi, mj))

print("pure python")
get_ipython().magic('timeit pyargmaxsq(pysq)')
print("numpy")
get_ipython().magic('timeit npargmaxsq(npsq)')



