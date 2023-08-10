import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('pylab', 'inline')
#%qtconsole
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use('ggplot')
np.random.seed(10)

N  = 1000
x  = np.linspace(0,1,N)
r  = stats.arcsine.rvs(size=N)

def arcsinepdf(x):
    return 1/(np.pi*np.sqrt(x*(1-x)))
def arcsinecdf(x):
    return 2/np.pi*np.arcsin(np.sqrt(x))

plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(1, 2)

ax0 = plt.subplot(gs[0])
ax0.plot(x, stats.arcsine.pdf(x),
         'r-', lw=5, alpha=0.5, label='arcsine pdf')
ax0.plot(x, arcsinepdf(x), 'k-', lw=1)
ax0.hist(r, normed=True, histtype='step', cumulative=False,
         alpha=0.5,bins=100,lw=2,color='b')
    
ax1 = plt.subplot(gs[1])
ax1.plot(x, stats.arcsine.cdf(x),
         'r-', lw=5, alpha=0.5, label='arcsine cdf')
ax1.plot(x, arcsinecdf(x), 'k-', lw=1)
ax1.hist(r, normed=True, histtype='step', cumulative=True,
         alpha=0.5,bins=100,lw=2,color='b')
plt.show()

Nprocess = 1000

Wiener = np.cumsum(np.random.randn(Nprocess,N),axis=0)
Wiener[0,:]=0
plt.figure(figsize=(18,6))
plt.plot(x,Wiener[:,:100],alpha=.5)
plt.plot(x,np.zeros(shape=x.shape),color='k',lw=1)
plt.show()

timediff = np.hstack([0,np.diff(x)])
Tplus    = np.zeros(N)
for iw in range(Nprocess):
    Wiener_it = Wiener[:,iw]
    Tplus[iw] = np.sum(timediff[Wiener_it>0])

plt.figure(figsize=(18,6))
gs2 = gridspec.GridSpec(1, 5)

ax0 = plt.subplot(gs2[0,0:3])
ax0.plot(x,Wiener[:,:100],color='gray',alpha=.5)
ax0.plot(x,Wiener[:,-1], color='b')
ax0.plot(x,np.zeros(shape=x.shape),color='k',lw=1)
ax0.plot(x[Wiener[:,-1]>0],Wiener[Wiener[:,-1]>0,-1],'r.')

ax1 = plt.subplot(gs2[0,3:5])
ax1.plot(x, arcsinecdf(x), 'k-', lw=1)
ax1.hist(Tplus, normed=True, histtype='step', cumulative=True,
         alpha=0.5,bins=100,lw=2,color='b')
         
plt.show()

L    = np.zeros(N,dtype=int)
for iw in range(Nprocess):
    Wiener_it = Wiener[:,iw]
    signchange= np.hstack([0,np.diff(np.sign(Wiener_it))])!=0
    L[iw]     = np.where(signchange)[0][-1]

plt.figure(figsize=(18,6))

ax0 = plt.subplot(gs2[0,0:3])
ax0.plot(x,Wiener[:,:100],color='gray',alpha=.5)
ax0.plot(x,np.zeros(shape=x.shape),color='k',lw=1)
ax0.plot(x,Wiener[:,-1], color='b')
ax0.plot(x[signchange],Wiener[signchange,-1],'g.',lw = 5)
ax0.plot(x[L[-1]],Wiener[L[-1],-1],'r.',lw = 5)

ax1 = plt.subplot(gs2[0,3:5])
ax1.plot(x, arcsinecdf(x), 'k-', lw=1)
ax1.hist(x[L], normed=True, histtype='step', cumulative=True,
         alpha=0.5,bins=100,lw=2,color='b')
         
plt.show()

W    = np.zeros(N,dtype=int)
for iw in range(Nprocess):
    Wiener_it = Wiener[:,iw]
    W[iw]     = Wiener_it.argmax()

plt.figure(figsize=(18,6))

ax0 = plt.subplot(gs2[0,0:3])
ax0.plot(x,Wiener[:,:100],color='gray',alpha=.5)
ax0.plot(x,np.zeros(shape=x.shape),color='k',lw=1)
ax0.plot(x,Wiener[:,-1], color='b')
ax0.plot(x[W[-1]],Wiener[W[-1],-1],'r.',lw = 5)

ax1 = plt.subplot(gs2[0,3:5])
ax1.plot(x, arcsinecdf(x), 'k-', lw=1)
ax1.hist(x[W], normed=True, histtype='step', cumulative=True,
         alpha=0.5,bins=100,lw=2,color='b')
         
plt.show()

