import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
 
with open( '../data/ZoneA.dat', 'r') as f:
    z = f.readlines()
z = [ i.strip().split() for i in z[10:] ]
z = np.array( z, dtype=np.float )
z = pd.DataFrame( z, columns=['x', 'y', 'thk', 'por', 'perm', 'lperm', 'lpermp', 'lpermr'] )

fig, ax = plt.subplots()
ax.scatter( z.x, z.y, c=z.por)
ax.set_aspect(1)
plt.xlim(-1500,22000)
plt.ylim(-1500,17500)
plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('Porosity %')
plt.show()

def SVh(P, h, bw):
    '''
    Experimental semivariogram for a single lag.
    '''
    dists = squareform(pdist(P[:,:2]))
    N = dists.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( dists[i,j] >= h-bw )and( dists[i,j] <= h+bw ):
                Z.append(( P[i,2] - P[j,2])**2)
    return np.sum(Z) / (2.0 * len( Z ))
 
def SV(P, hs, bw):
    '''
    Experimental variogram for a collection of lags.
    '''
    sv = list()
    for h in hs:
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T

def C(P, h, bw):
    '''
    Calculate the sill.
    '''
    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )

# Part of our data set recording porosity.
P = np.array(z[['x', 'y', 'por']])

# Bandwidth, plus or minus 250 meters.
bw = 500

# Lags in 500 meter increments from zero to 10,000.
hs = np.arange(0, 10500, bw)
sv = SV( P, hs, bw )

# Make a plot.
plt.plot( sv[0], sv[1], '.-' )
plt.xlabel('Lag [m]')
plt.ylabel('Semivariance')
plt.title('Sample semivariogram') ;
plt.show()

def opt(func, x, y, C0, parameterRange=None, meshSize=1000):
    if parameterRange == None:
        parameterRange = [x[1], x[-1]]
    mse = np.zeros(meshSize)
    a = np.linspace(parameterRange[0], parameterRange[1], meshSize)
    for i in range(meshSize):
        mse[i] = np.mean((y - func(x, a[i], C0))**2.0)
    return a[mse.argmin()]

def spherical(h, a, C0):
    '''
    Spherical model of the semivariogram
    '''
    # If h is a scalar:
    if np.ndim(h) == 0:
        # Calculate the spherical function.
        if h <= a:
            return C0 * ( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    else:
        # Calculate the spherical function for all elements.
        a = np.ones(h.size) * a
        C0 = np.ones(h.size) * C0
        return np.array(list(map(spherical, h, a, C0)))

def cvmodel(P, model, hs, bw):
    '''
    Input:  (P)      ndarray, data
            (model)  modeling function
                      - spherical
                      - exponential
                      - gaussian
            (hs)     distances
            (bw)     bandwidth
    Output: (covfct) function modeling the covariance
    '''
    # Calculate the semivariogram.
    sv = SV(P, hs, bw)
    # Calculate the sill.
    C0 = C(P, hs[0], bw)
    # Calculate the optimal parameters.
    param = opt(model, sv[0], sv[1], C0)
    # Return a covariance function.
    return lambda h, a=param: C0 - model(h, a, C0)

sp = cvmodel(P, model=spherical, hs=np.arange(0, 10500, 500), bw=500)

plt.plot( sv[0], sv[1], '.-' )
plt.plot( sv[0], sp( sv[0] ) ) ;
plt.title('Spherical Model')
plt.ylabel('Semivariance')
plt.xlabel('Lag [m]')
plt.show()

def krige(P, model, hs, bw, u, N):
    '''
    Input  (P)     ndarray, data
           (model) modeling function
                    - spherical
                    - exponential
                    - gaussian
           (hs)    kriging distances
           (bw)    kriging bandwidth
           (u)     unsampled point
           (N)     number of neighboring
                   points to consider
    '''
 
    # covariance function
    covfct = cvmodel(P, model, hs, bw)
    # mean of the variable
    mu = np.mean(P[:,2])
 
    # distance between u and each data point in P
    d = np.sqrt((P[:,0]-u[0])**2.0 + (P[:,1]-u[1])**2.0)
    # add these distances to P
    P = np.vstack(( P.T, d )).T
    # sort P by these distances
    # take the first N of them
    P = P[d.argsort()[:N]]
 
    # apply the covariance model to the distances
    k = covfct( P[:,3] )
    # cast as a matrix
    k = np.matrix( k ).T
 
    # form a matrix of distances between existing data points
    K = squareform( pdist( P[:,:2] ) )
    # apply the covariance model to these distances
    K = covfct( K.ravel() )
    # re-cast as a NumPy array -- thanks M.L.
    K = np.array( K )
    # reshape into an array
    K = K.reshape(N,N)
    # cast as a matrix
    K = np.matrix( K )
 
    # calculate the kriging weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )
 
    # calculate the residuals
    residuals = P[:,2] - mu
 
    # calculate the estimation
    estimation = np.dot( weights.T, residuals ) + mu
 
    return float( estimation )

P[:,0].min(), P[:,0].max(), P[:,1].min(), P[:,1].max()

X0, X1 = 0, 20000
Y0, Y1 = 0, 16000

# Define the number of grid cells over which to make estimates.
# TODO: Vectorize this. I'll try numba/jit but I don't think it'll help.
# I think it can be vectorized with np.mgrid (better than np.meshgrid)

# Many points:
x, y = 100, 80

# Fewer points:
x, y = 50, 40

dx, dy = (X1-X0) / x, (Y1-Y0) / y

def stepwise(x, y):
    Z = np.zeros((y, x))
    
    for i in range(y):
        print(i, end=' ')
        for j in range(x):
            Z[i, j] = krige(P, model=spherical, hs=hs, bw=bw, u=(dy*j, dx*i), N=16)
            
    return Z

# THIS IS SLOW
# Z = stepwise(x, y)

Z

extent = [X0, X1, Y0, Y1]

plt.imshow(Z, origin='lower', interpolation='none', extent=extent)
plt.scatter(z.x, z.y, s=2, c='w')
 
plt.show()



