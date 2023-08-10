import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
sns.set_context(context='paper',font_scale=1.5)
get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#alter the line below to correspond to your file system
nonlinpy_dir = '/Users/nickc/Documents/skccm'
sys.path.append(nonlinpy_dir)

import skccm.paper as paper
import skccm.data as data

rx1 = 3.72 #determines chaotic behavior of the x1 series
rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = 0.2 #Influence of x1 on x2
b21 = 0.01 #Influence of x2 on x1
ts_length = 1000
x1,x2 = data.coupled_logistic(rx1,rx2,b12,b21,ts_length)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(x1[0:100])
ax[1].plot(x2[0:100])
ax[0].set_yticks([.1,.3,.5,.7,.9])
ax[1].set_xlabel('Time')
sns.despine()

#fig.savefig('../figures/train_test_split/coupled_logistic.png',bbox_inches='tight')

e1 = paper.Embed(x1)
e2 = paper.Embed(x2)

mi1 = e1.mutual_information(10)
mi2 = e2.mutual_information(10)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(np.arange(1,11),mi1)
ax[1].plot(np.arange(1,11),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

#fig.savefig('../figures/mutual_info.png',bbox_inches='tight')

lag = 1
embed = 2
X1 = e1.embed_vectors_1d(lag,embed)
X2 = e2.embed_vectors_1d(lag,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

#fig.savefig('../figures/x_embedded.png',bbox_inches='tight')

CCM = ccm.CCM()

from skccm.utilities import train_test_split

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

len_tr = len(x1tr)
lib_lens = np.linspace(10, len_tr/2, dtype='int')
lib_lens

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens)

sc1,sc2 = CCM.score()

fig,ax = plt.subplots()
ax.plot(lib_lens,sc1,label='X1 xmap X2')
ax.plot(lib_lens,sc2, label='X2 xmap X1')
ax.set_xlabel('Library Length')
ax.set_ylabel('Forecast Skill')
sns.despine()

#fig.savefig('../figures/train_test_split/xmap_lib_len.png',bbox_inches='tight')

rx1 = 3.7 #determines chaotic behavior of the x1 series
rx2 = 3.7 #determines chaotic behavior of the x2 series

ts_length = 1000

#variables for embedding
lag = 1
embed = 2

CCM = ccm.CCM() #intitiate the ccm object

#store values
num_b = 20 #number of bs to test
sc1_store = np.empty((num_b,num_b))
sc2_store = np.empty((num_b,num_b))

#values over which to test
max_b = .4 #maximum b values
b_range = np.linspace(0,max_b,num=num_b)

#loop through b values for b12 and b21
for ii,b12 in enumerate(b_range):
    for jj, b21 in enumerate(b_range):
        
        x1,x2 = data.coupled_logistic(rx1,rx2,b12,b21,ts_length)
        
        e1 = ccm.Embed(x1)
        e2 = ccm.Embed(x2)
        
        X1 = e1.embed_vectors_1d(lag,embed)
        X2 = e2.embed_vectors_1d(lag,embed)
        
        x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)
        
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=[500]) #only one prediction
        sc1,sc2 = CCM.score()
        
        
        sc1_store[ii,jj] = sc1[0]
        sc2_store[ii,jj] = sc2[0]
        
sc_diff = sc2_store-sc1_store

fig,ax = plt.subplots()
v = np.linspace(-1.6,1.6,21)
cax = ax.contourf(b_range,b_range,sc_diff,v,cmap='magma')
fig.colorbar(cax,ticks=v[::2])
ax.set_xlabel('B12')
ax.set_ylabel('B21')

#fig.savefig('../figures/train_test_split/xmap_changingB.png',bbox_inches='tight')

rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = .2 #Influence of x1 on x2

ts_length = 1000
x1,x2 = data.driven_rand_logistic(rx2,b12,ts_length)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(x1[0:100])
ax[1].plot(x2[0:100])
ax[0].set_yticks([.1,.3,.5,.7,.9])
ax[1].set_xlabel('Time')
sns.despine()

e1 = ccm.Embed(x1)
e2 = ccm.Embed(x2)

mi1 = e1.mutual_information(10)
mi2 = e2.mutual_information(10)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(np.arange(1,11),mi1)
ax[1].plot(np.arange(1,11),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

lag = 1
embed = 3

X1 = e1.embed_vectors_1d(lag,embed)
X2 = e2.embed_vectors_1d(lag,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

CCM = ccm.CCM()

lib_lens = np.linspace(10,ts_length/2,dtype='int')

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens) #only one prediction
sc1,sc2 = CCM.score()

plt.plot(sc1)
plt.plot(sc2)

rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = .5 #Influence of x1 on x2

ts_length = 1000
x1,x2 = data.driving_sin(rx2,b12,ts_length)

fig,ax = plt.subplots(nrows=2,sharex=True)
ax[0].plot(x1[0:100])
ax[1].plot(x2[0:100])
ax[1].set_xlabel('Time')
sns.despine()

em_x1 = ccm.Embed(x1)
em_x2 = ccm.Embed(x2)

mi1 = em_x1.mutual_information(10)
mi2 = em_x2.mutual_information(10)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(np.arange(1,11),mi1)
ax[1].plot(np.arange(1,11),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

lag1 = 4
lag2 = 1
embed = 3

X1 = em_x1.embed_vectors_1d(lag1,embed)
X2 = em_x2.embed_vectors_1d(lag2,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

CCM = ccm.CCM()

print('X1 Shape:', X1.shape)
print('X2 Shape:', X2.shape)

X2 = X2[0:len(X1)]

lib_lens = np.linspace(10,ts_length/2,dtype='int')

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens) #only one prediction
sc1,sc2 = CCM.score()

plt.plot(sc1)
plt.plot(sc2)

x1p[-1].shape

x1te.shape

plt.plot(x1p[-1][:,0])
plt.plot(x1te[:,0])

plt.plot(x2te[:,0],alpha=.5)
plt.plot(x2p[-1][:,0])

rx1 = 3.72 #determines chaotic behavior of the x1 series
rx2 = 3.72 #determines chaotic behavior of the x2 series
b12 = 0.01 #Influence of x1 on x2
b21 = 0.3 #Influence of x2 on x1
ts_length = 8000
x1,x2 = data.lagged_coupled_logistic(rx1,rx2,b12,b21,ts_length)

fig,ax = plt.subplots(nrows=2,sharex=True)
ax[0].plot(x1[0:100])
ax[1].plot(x2[0:100])
ax[1].set_xlabel('Time')
sns.despine()

em_x1 = ccm.Embed(x1)
em_x2 = ccm.Embed(x2)

mi1 = em_x1.mutual_information(10)
mi2 = em_x2.mutual_information(10)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(np.arange(1,11),mi1)
ax[1].plot(np.arange(1,11),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

lag1 = 2
lag2 = 2
embed = 3

X1 = em_x1.embed_vectors_1d(lag1,embed)
X2 = em_x2.embed_vectors_1d(lag2,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

CCM = ccm.CCM()

lib_lens = np.linspace(10,ts_length/2,dtype='int')

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens) #only one prediction
sc1,sc2 = CCM.score()

plt.plot(lib_lens,sc1)
plt.plot(lib_lens,sc2)



ts_length=1000
x1 = np.random.randn(ts_length,) + np.linspace(0,25,ts_length)
x2 = np.random.randn(ts_length,) + np.linspace(0,10,ts_length)

plt.plot(x1)
plt.plot(x2)

em_x1 = ccm.Embed(x1)
em_x2 = ccm.Embed(x2)

mi1 = em_x1.mutual_information(20)
mi2 = em_x2.mutual_information(20)

fig,ax = plt.subplots(nrows=2,sharex=True)
ax[0].plot(np.arange(1,21),mi1)
ax[1].plot(np.arange(1,21),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

lag1 = 2
lag2 = 2
embed = 8

X1 = em_x1.embed_vectors_1d(lag1,embed)
X2 = em_x2.embed_vectors_1d(lag2,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

CCM = ccm.CCM()

lib_lens = np.linspace(10,ts_length/2,dtype='int')

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens) #only one prediction
sc1,sc2 = CCM.score()

plt.plot(lib_lens,sc1)
plt.plot(lib_lens,sc2)

ts_length=500
x1 = np.random.randn(ts_length,) + 10*np.sin(np.linspace(0,10*np.pi,ts_length))
x2 = np.random.randn(ts_length,) + 20*np.cos(np.linspace(0,10*np.pi,ts_length))

#x1[x1<0] = np.random.randn(np.sum(x1<0),)
#x2[x2<0] = np.random.randn(np.sum(x2<0),)

plt.plot(x1)
plt.plot(x2,alpha=.5)

em_x1 = ccm.Embed(x1)
em_x2 = ccm.Embed(x2)
mi1 = em_x1.mutual_information(100)
mi2 = em_x2.mutual_information(100)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(np.arange(1,101),mi1)
ax[1].plot(np.arange(1,101),mi2)
ax[1].set_xlabel('Lag')
sns.despine()

lag1 = 20
lag2 = 20
embed = 2

X1 = em_x1.embed_vectors_1d(lag1,embed)
X2 = em_x2.embed_vectors_1d(lag2,embed)

fig,ax = plt.subplots(ncols=2,sharey=True,sharex=True,figsize=(10,4)) 
ax[0].scatter(X1[:,0],X1[:,1])
ax[1].scatter(X2[:,0],X2[:,1])
ax[0].set_xlabel('X1(t)')
ax[0].set_ylabel('X1(t-1)')
ax[1].set_xlabel('X2(t)')
ax[1].set_ylabel('X2(t-1)')
sns.despine()

CCM = ccm.CCM()

lib_lens = np.linspace(10,ts_length/2,dtype='int')

x1tr, x1te, x2tr, x2te = train_test_split(X1,X2, percent=.75)

CCM.fit(x1tr,x2tr)
x1p, x2p = CCM.predict(x1te, x2te,lib_lengths=lib_lens) #only one prediction
sc1,sc2 = CCM.score()

plt.plot(lib_lens,sc1)
plt.plot(lib_lens,sc2)

