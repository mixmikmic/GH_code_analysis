from numpy import linspace,zeros
from numpy.linalg import cond

for N in range(1,25):
    x = linspace(-1.0,+1.0,N+1)
    V = zeros((N+1,N+1))
    for j in range(0,N+1):
        V[j][:] = x**j
    print "%8d %20.8e" % (N,cond(V.T))

