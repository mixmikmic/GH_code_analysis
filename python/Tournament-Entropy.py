import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import eigen
import utils

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

N = 1000
V = 10
B = 0.9

X = utils.gen_data_pr(V, B, N)
R = utils.get_interactions(X).astype(float)
R

R.sum(axis=0) / R.sum()

eigen.te(R)

eigen.te_max(V)

N = 1000
V = 10
B = 0.9
X = utils.gen_data_pr(V, B, N)
R = utils.get_interactions(X).astype(float)
eigen.te(R)

N = 1000
V = 10
B_vals = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]

results = []
for B in B_vals:
    X = utils.gen_data_pr(V, B, N)
    R = utils.get_interactions(X).astype(float)
    results.append(eigen.te(R))

plt.plot(B_vals, results)

N = 10000
B = .8
V_vals = [5, 10, 50, 100]

results = []
for V in V_vals:
    X = utils.gen_data_pr(V, B, N)
    R = utils.get_interactions(X).astype(float)
    results.append(eigen.te(R))

plt.plot(V_vals, results)

R = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]])
R_nrm = eigen.get_R_nrm(R)
print eigen.te(R)
vec = eigen.power_method(R_nrm)
val, vec = np.linalg.eig(R_nrm.T)
val

vec

R = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0]])
R_nrm = eigen.get_R_nrm(R)
print eigen.te(R)
vec = eigen.power_method(R_nrm)
print '||v1||:', np.linalg.norm(vec)**2
vec

R = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0]])
R_nrm = eigen.get_R_nrm(R)
print eigen.te(R)
vec = eigen.power_method(R_nrm)
print '||v1||:', np.linalg.norm(vec)**2
vec

R = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0]])
R_nrm = eigen.get_R_nrm(R)
print eigen.te(R)
vec = eigen.power_method(R_nrm)
print '||v1||:', np.linalg.norm(vec)**2
R_nrm

R = np.array([
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0]])
R_nrm = eigen.get_R_nrm(R)
print eigen.te(R)
vec = eigen.power_method(R_nrm)
print '||v1||:', np.linalg.norm(vec)**2
vec

R = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]])
R_nrm = eigen.get_R_nrm(R)
print 'TE:', eigen.te(R)
vec = eigen.power_method(R_nrm)
print '||v1||:', np.linalg.norm(vec)**2
vec



