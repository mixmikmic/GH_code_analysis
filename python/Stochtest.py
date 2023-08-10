## Here is some simple code just to show how to get a brownian motion path.
## From the book Python for Scientists by John Stewart.

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

W0 = 0  # the initial point of the path
V0 = 1  # the variance scale
T=1   # time interval of one second
N = 500  # number of steps in the interval
t,dt=np.linspace(0,T,N+1,retstep=True)
dW=npr.normal(0.0,np.sqrt(V0*dt),N+1)   # the delta W in the Brownian motion
dW[0]=W0   # set initial position to W0
W=np.cumsum(dW)   # integrate the dW

plt.ion()  # interactive plot mode

plt.plot(t,W)
plt.xlabel('t')
plt.ylabel('W(t)')
plt.title('Sample Wiener Process',weight='bold',size=16)

def weiner(W0,V0,T,N):
    t,dt=np.linspace(0,T,N+1,retstep=True)
    dW=npr.normal(0.0,np.sqrt(V0*dt),N+1)   # the delta W in the Brownian motion
    dW[0]=W0   # set initial position to W0
    W=np.cumsum(dW)   # integrate the dW
    return t,W

t,W = weiner(1,1,1,500)
plt.plot(t,W)

    

np.size(W)

W[500]

np.average(W)

# So, let's run a bunch of trials of the brownian motion.
# We will only look at the last value in the path
# The average of these should be W0
# The variance of these should be V0*T

W0=1.234
V0=.37
T=2
nsteps = 500

ntrials=10000
results = np.zeros(ntrials)

for k in range(ntrials):
    t,w = weiner(W0,V0,T,nsteps)
    results[k]=w[-1]
   
np.average(results),np.var(results)/T
    


# set some parameters for the Weiner process
W0=0
V0=1
T=10
nsteps = 1000
ntrials = 5000
t,dt = np.linspace(0,T,nsteps+1,retstep=True)
dW = npr.normal(0.0,np.sqrt(V0*dt),(ntrials,nsteps+1)) # this is a 2D array of normal values
dW[:,0]=W0
W=np.cumsum(dW,axis=1)  # these are the Brownian motion paths
Y=np.exp(-0.5*W)  # now we take their exponentials, with the coefficient 1/2

Ymean = np.mean(Y,axis=0)  # this is the average of all our paths
Yexact = np.exp(t/8)

# We plot a few curves
plt.plot(t,Y[0,:],t,Y[1,:],t,Y[2,:],t,Y[3,:],t,Y[4,:])

plt.plot(t,Ymean,t,Yexact)

## Oh heck, let's get everything just the way I want it

# set some parameters for the Weiner process
W0=0
V0=1
T=1
nsteps = 1000
ntrials = 10000
t,dt = np.linspace(0,T,nsteps+1,retstep=True)
dW = npr.normal(0.0,np.sqrt(V0*dt),(ntrials,nsteps+1)) # this is a 2D array of normal values
dW[:,0]=W0
W=np.cumsum(dW,axis=1)  # these are the Brownian motion paths
Y=np.exp(-0.5*W)  # now we take their exponentials, with the coefficient 1/2

Ymean = np.mean(Y,axis=0)  # this is the average of all our paths
Yexact = np.exp(t/8)

# we plot
plt.plot(t,Ymean,t,Yexact)
plt.title("Number of trials = %d " % ntrials)



