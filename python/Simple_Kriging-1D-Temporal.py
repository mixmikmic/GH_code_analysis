from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [16 / 1.5, 10 / 1.5]   # inch / cm = 2.54
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
# plt.rcParams['savefig.frameon'] = False

import pandas as pd
df_Qts = pd.read_csv('5297Q_2010-14.txt',
                usecols =["DATUM","MESSWERT_NUM"] ,
                index_col=0, 
                parse_dates=True, 
                infer_datetime_format=True,
                dayfirst=True,
                decimal='.',
                sep=",")

#plt.plot(df_5297T, label=u"5297T (Tertiär)")
plt.plot(df_Qts, label=u"5297Q (Quartär)")
plt.ylabel(u"Grundwasserstand\n[mNN]")
plt.legend()

#transform date to float and give differnce in days
import pandas as pd
import numpy as np

X = np.array(df_Qts.index.values - df_Qts.index.values.min(), dtype=(float)) / (1e9 * 60 * 60 * 24)

df_Qts = df_Qts.assign(X =X [:])
#df_QtsX= pd.write_csv('5297QX_2010-11.txt')({'DATUM' :df_Qts.index.values , 'MESSWERT_NUM' : df_Qts })

df_Qts["Y"] = 0.
P = np.array( df_Qts.dropna()[['X','Y','MESSWERT_NUM']] )
P


def SVh( P, h, bw ):
    '''
    Experimental semivariogram for a single lag
    '''
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )  # sample difference
    return np.sum( Z ) / ( 2.0 * len( Z ) )
 
def SV( P, hs, bw ):
    '''
    Experimental variogram for a collection of lags
    '''
    sv = list()
    for h in hs:
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T
 
def C( P, h, bw ):
    '''
    Calculate the sill
    '''
    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )


# bandwidth, plus or minus bw meters
bw = 25
# lags in 25 days increments from zero to 2000
hs = np.arange(0, 2000, bw)
sv = SV( P, hs, bw )
plot( sv[0], sv[1], '.-' )
xlabel('Lag [m]')
ylabel('Semivariance')
title('Sample Semivariogram') ;
savefig('sample_semivariogram.png',fmt='png',dpi=200)

def opt( fct, x, y, C0, parameterRange=None, meshSize=1000 ):
    if parameterRange == None:
        parameterRange = [ x[1], x[-1] ]
    mse = np.zeros( meshSize )
    a = np.linspace( parameterRange[0], parameterRange[1], meshSize )
    for i in range( meshSize ):
        mse[i] = np.mean( ( y - fct( x, a[i], C0 ) )**2.0 )
    return a[ mse.argmin() ]

def gaussian( h, a, C0, Cn=0 ):
    '''
    Gaussian model of the semivariogram
    '''
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        return Cn+(C0-Cn) * (1 - exp(-3*h**2/a**2))
        
    # if h is an iterable
    else:
        # calcualte the gaussian function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( gaussian, h, a, C0, Cn )

def exponential( h, a, C0, Cn=0 ):
    '''
    Exponential model of the semivariogram
    '''
    # if h is a single digit
    if type(h) == np.float64:
        
        # calculate the exponential function
        return Cn+(C0-Cn) * (1 - exp(-3*h/a))
        
    # if h is an iterable
    else:
        # calcualte the exponential function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( exponential, h, a, C0, Cn )

def spherical( h, a, C0, Cn=0 ):
    '''
    Spherical model of the semivariogram
    '''

    if type(C0) == float:
        C0 = np.float64(C0)
        
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        if h <= a:
            return Cn+(C0-Cn)*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    # if h is an iterable
    else:
        # calcualte the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones( h.size ) * Cn
        return map( spherical, h, a, C0, Cn )

def hole (h, a, C0, Cn=0):
    
    if type(h) == np.float64:
        # calculate the spherical function
        return C0*(1-(1-h/a) * exp(-h/a) )

    # if h is an iterable
    else:
        # calcualte the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( hole, h, a, C0, Cn )

def hole_N (h, C0, a, Cn=0):
    #from Triki et al. p.1600 (Dowdall et al. 2003)
    if type(h) == np.float64:
        # calculate the hole function
        if h == 0:
            return Cn
        if h <= pi*2*a: 
            return (Cn+(C0-Cn)*(1-(sin(h/a ))/(h/a) ))
        if pi*2*a < h <= pi*4*a:
            return (Cn+(C0-Cn)*(1-(sin(h/a ))/(0.6*h/a) ))
        if h > pi*4*a:
            return (Cn+(C0-Cn)*(1-(sin(h/a ))/(0.4*h/a) ))
    # if h is an iterable
    else:
        # calcualte the hole function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        Cn = np.ones(h.size) * Cn
        return map( hole_N, h, a, C0, Cn )

def cvmodel( P, model, hs, bw, Cn = None, svrange=None, C0=None):
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
    if Cn is None:
        Cn = N(P, hs, bw)
        
    if type(C0) == float:
        C0 = np.float64(C0)
        
    # calculate the semivariogram
    sv = SV( P, hs, bw )
    # calculate the sill
    if C0 is None:
        C0 = C( P, hs[0], bw )
    # calculate the optimal parameters
    if svrange is None:
        svrange = opt( model, sv[0], sv[1], C0 )
    # return a covariance function
    covfct = lambda h, a=svrange: C0 - model( h, a, C0, Cn=Cn )
    return covfct

varmodel = hole_N
C0=0.05
sp = cvmodel( P, model=varmodel, hs=np.arange(0,2000,25), bw=bw, Cn=0.0, svrange=365/(2*pi), C0=C0)
#C0 = C( P, hs[0], bw )

plot( sv[0], sv[1], '.-' )
plot( sv[0], C0 - sp( sv[0] ) ) ;
title('Hole_N Model')
ylabel('Semivariance')
xlabel('Lag [m]')
savefig('semivariogram_model_hole_N_highamplitude2.png',fmt='png',dpi=300)

def krige( P, covfct, u, N ):
    '''
    Input  (P)     ndarray, data
           (covfct) modeling function
                    - spherical
                    - exponential
                    - gaussian
           (u)     unsampled point
           (N)     number of neighboring
                   points to consider
    '''

    assert N < len(P) + 1, "Number of Neighbors greater than number of data points"

    # mean of the variable
    mu = np.mean( P[:,2] )
 
    # distance between u and each data point in P
    d = np.sqrt( ( P[:,0]-u[0] )**2.0 + ( P[:,1]-u[1] )**2.0 )
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
    K = np.identity(40)*0.025+K
 
    # calculate the kriging weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )
 
    # calculate the residuals
    residuals = P[:,2] - mu
 
    # calculate the estimation
    estimation = np.dot( weights.T, residuals ) + mu
 
    return (float( estimation ), k, K, weights)

C0-np.dot(weights,k.T)

# sampling intervall:
X0, X1 = 0., 3000.

# number of samples
n = 1800
dx = (X1-X0)/n
print("Sampling resolution: {:.2f} days".format(dx))

# number of neighbors:
nn = 40
u = (1800, 0)  # (x,y) - coordinates of unsampled points

# create a progress bar
from ipywidgets import FloatProgress
from IPython.display import display
wdgt = FloatProgress(min=0, max=n, description="Running Kriging ... ")
display(wdgt)

Z = np.zeros(n)
V = np.zeros(n)
for i in range(n):
    wdgt.value += 1
    h, k, K, weights = krige( P, sp, (dx*i, 0), nn )
    Z[i] = h
    v = C0 - np.dot(k.T, weights)
    V[i] = max (v, 0)


df_Qts14 = pd.read_csv('5297Q_2014-16.txt',
                usecols =["DATUM","MESSWERT_NUM"] ,
                index_col=0, 
                parse_dates=True, 
                infer_datetime_format=True,
                dayfirst=True,
                decimal='.',
                sep=",")

X14 = np.array(df_Qts14.index.values - df_Qts14.index.values.min() , dtype=(float)) / (1e9 * 60 * 60 * 24)

df_Qts14 = df_Qts14.assign(X =X14[:])
#df_QtsX= pd.write_csv('5297QX_2010-11.txt')({'DATUM' :df_Qts.index.values , 'MESSWERT_NUM' : df_Qts })

df_Qts14["Y"] = 0.
P14 = np.array( df_Qts14.dropna().reset_index()[['DATUM','Y','MESSWERT_NUM']] )

P14

plt.rcParams["figure.figsize"] = (16, 12)
plt.fill_between([i*dx for i in range(n)], Z-3*np.sqrt(V), Z+3*np.sqrt(V), color='green', alpha=0.1, label="+/- 3 stdev")
plt.fill_between([i*dx for i in range(n)], Z-2*np.sqrt(V), Z+2*np.sqrt(V), color='green', alpha=0.2, label="+/- 2 stdev")
plt.fill_between([i*dx for i in range(n)], Z-1*np.sqrt(V), Z+1*np.sqrt(V), color='green', alpha=0.3, label="+/- 1 stdev")
plt.plot([i*dx for i in range(n)], Z, "g-", label="estimator")
plt.scatter(df_Qts.X, df_Qts.MESSWERT_NUM, c="r", label="sample", s=3)
plt.plot((X0, X1),(df_Qts.MESSWERT_NUM.mean(),df_Qts.MESSWERT_NUM.mean()), "r-.", label="mean of sample", alpha=0.3,)
plt.plot(range(1612,1612+len(df_Qts14.MESSWERT_NUM)), df_Qts14.MESSWERT_NUM)
plt.legend(loc='upper right')
ylabel('Water Level [m]')
xlabel('Time [d]')

x1,x2,y1,y2 = plt.axis()

plt.axis((x1,x2,447, 450))

plt.savefig("result_hole_N_model_highAmplitude_SKwithtimeseries40nn.png")
plt.show()

plt.plot([i*dx for i in range(n)], 2*1.96*np.sqrt(V), "g--", label="size of estimated 95% confidence intervall")
plt.plot((X0, X1),(2*1.96*np.sqrt(C0), 2*1.96*np.sqrt(C0)), "r-", label="size of confidence from global Sill")
plt.legend(loc='upper left')
#plt.scatter(df_Qts.X, (df_Qts.MESSWERT_NUM-448)*30, c="r", label="sample", s=3)

#plt.scatter(df_Qts.X, np.ones_like(df_Qts.X), c="r", label="sample", s=3)

ylabel('Value [m]')
xlabel('Time [d]')
#plt.axis((-1,10,0, 25))
plt.savefig("result_hole_N_model_highAmplitude_95%interval40nn.png")
_ = plt.show()

get_ipython().magic('matplotlib inline')

sqrt(0.02)



