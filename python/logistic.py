get_ipython().run_line_magic('matplotlib', 'inline')
from pymc3 import *
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

d = np.genfromtxt('data/phonemes.csv', delimiter=',')

d = np.delete(d, (0), axis = 0)
X = d[:,range(d.shape[1]-1)]
y = d[:,d.shape[1]-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66, random_state=108)

with Model() as model:
    beta = pm.Normal('beta', 0, sd = 1, shape = 256)
    alpha = pm.Normal('alpha', 0, sd = .01)
    p = pm.invlogit(alpha + pm.math.dot(X_train, beta))
    out = pm.Bernoulli('y', p= p, observed = y_train, total_size=y_train.shape[0])    
    
with model:
    start = pm.find_MAP(model = model)
    t1 = pm.sample(1000, njobs=1, start = start)

plt.figure(figsize=(15, 5));
plt.plot(np.arange(256), t1[500:]['beta'].mean(axis=0));

pm.plots.traceplot(t1);

# Scaling first 
X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=108)

with Model() as model:
    beta = pm.Normal('beta', 0, sd = 1, shape = 256)
    alpha = pm.Normal('alpha', 0, sd = .01)
    p = pm.invlogit(alpha + pm.math.dot(X_train, beta))
    
    out = pm.Bernoulli('y', p= p, observed = y_train, total_size=y_train.shape[0])  
    
with model:
    start = pm.find_MAP(model = model)
    t4 = pm.sample(1000, njobs=1, start = start)

plt.figure(figsize=(15, 5));
plt.plot(np.arange(256), t4[500:]['beta'].mean(axis=0));

pm.plots.traceplot(t4);

with Model() as model:
    beta = pm.Normal('beta', 0, sd = .01, shape = 256)
    alpha = pm.Normal('alpha', 0, sd = .01)
    p = pm.invlogit(alpha + pm.math.dot(X_train, beta))
    out = pm.Bernoulli('y', p= p, observed = y_train, total_size=y_train.shape[0])    
    
with model:    
    start = pm.find_MAP(model = model)
    t5 = pm.sample(1000, njobs=1, start = start)

plt.figure(figsize=(15, 5));
plt.plot(np.arange(256), t4[500:]['beta'].mean(axis=0));

pm.plots.traceplot(t5);

X = d[:,range(d.shape[1]-1)]
y = d[:,d.shape[1]-1]

H = np.genfromtxt('data/spline_basis.csv', delimiter=',')
H = np.delete(H, (0), axis = 0)

Xp = np.matmul(X,H)
Xp[0,]

Xp = np.genfromtxt('data/spline_data.csv', delimiter=',')
Xp = np.delete(Xp, (0), axis = 0)

Xp_train, Xp_test, y_train, y_test = train_test_split(Xp, y, test_size=0.33, random_state=108)

with Model() as model:
    beta = pm.Normal('beta', 0, sd = 1e3, shape = 12)
    alpha = pm.Normal('alpha', 0, sd = 1e3)
    p = pm.invlogit(alpha + pm.math.dot(Xp_train, beta))
    
    out = pm.Bernoulli('y', p= p, observed = y_train, total_size=y_train.shape[0])  

with model:
    start = pm.find_MAP(model = model)
    t7 = pm.sample(2000, njobs=1, start = start)

pm.plots.traceplot(t7);

C = np.matmul(t7['beta'], H.T)
C = np.percentile(C, [2.5, 50, 97.5], axis = 0)
Cmap = np.matmul(start['beta'], H.T)
plt.figure(figsize=(15, 5));
plt.fill_between(np.arange(256), C[0], C[2], facecolor='gray', alpha = 0.4, zorder = 1);
plt.plot(np.arange(256), C[1,].T, label = 'Median', linestyle = '--', zorder = 2);
plt.plot(np.arange(256), Cmap.T);







