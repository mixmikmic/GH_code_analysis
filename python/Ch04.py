import math

import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# data resolution matrix

# z-axis
z = np.arange(101) / 10

# simulate data (true and, add noise to simulate observed)
dtrue = 2 + z
dobs = 2 + z + np.random.normal(0, 1, 101)

# build system matrix, solve for model estimate using least squares, calculate predicted data
G = np.array([np.ones(101), z])
GMG = inv(G.dot(G.transpose())).dot(G)
mest = GMG.dot(dobs)
dpre = G.transpose().dot(mest)

# calculate data resolution matrix
Nres = G.transpose().dot(GMG)

# build figure
plt.subplots(1, 2, figsize=(15, 5))

# simulated data with least squares fit
plt.subplot(121)
plt.scatter(z, dobs, color='red')
plt.plot(z, dtrue, color='black')
plt.plot(z, dpre, color='blue')
plt.xlim(0,10)
plt.ylim(0,12)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Simulated Data')

# data resolution matrix
plt.subplot(122)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Nres)
plt.imshow(Nres, interpolation='nearest')
plt.plot((0,100), (0,100), color='black', ls='dashed')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,100)
plt.ylim(100,0)
plt.xlabel('j')
plt.ylabel('i')
plt.title('Data Resolution Matrix')

# Dirichlet solution

# z-axis
z = np.arange(101) / 10

# build true model - mostly zero but a few spikes
mtrue = np.zeros(101)
mtrue[4] = 1
mtrue[9] = 1
mtrue[19] = 1
mtrue[49] = 1
mtrue[89] = 1

# experiment: exponential smoothing of model
c = np.arange(81) * 0.1
G = np.exp((-c).reshape(81,1) * z)
dtrue = G.dot(mtrue)
dobs = dtrue + np.random.normal(0, 1e-12, 81)

# minimum length solution
GMG = solve(G.dot(G.transpose()) + 1e-12 * np.eye(81), G).transpose()
mest = GMG.dot(dobs)
Rres = GMG.dot(G)

# build figure
plt.subplots(1, 2, figsize=(18, 5))

# true model parameters
plt.subplot(131)
plt.plot(z, mtrue, color='black')
plt.plot(z, mest, color='red')
plt.xlim(0,10)
plt.ylim(-0.2,1)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Dirichlet Approach')

# model resolution matrix
plt.subplot(132)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Rres)
plt.imshow(Rres, interpolation='nearest')
plt.plot((0,100), (0,100), color='black', ls='dashed')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,100)
plt.ylim(100,0)
plt.xlabel('j')
plt.ylabel('i')
plt.title('Model Resolution Matrix')

# least squares fit to two cases of synthetic data (distinguised by the spacing of the z's)

# CASE 1: data clumped in middle of interval
z1 = np.sort(np.random.uniform(4,6,10))
sd1 = 1
dobs1 = 5 + 0.5 * z1 + np.random.normal(0, sd1, 10)

# CASE 2: data spread out over whole interval
z2 = np.sort(np.random.uniform(0,10,10))
sd2 = 1
dobs2 = 5 + 0.5 * z2 + np.random.normal(0, sd2, 10)

# build system matrix and solve
G1 = np.array([np.ones(10), z1])
GTG_inv1 = inv(G1.dot(G1.transpose()))
mest1 = GTG_inv1.dot(G1.dot(dobs1))
Cm1 = sd1**2 * GTG_inv1
sm1 = np.power(Cm1.diagonal(), 0.5)
G2 = np.array([np.ones(10), z2])
GTG_inv2 = inv(G2.dot(G2.transpose()))
mest2 = GTG_inv2.dot(G2.dot(dobs2))
Cm2 = sd2**2 * GTG_inv2
sm2 = np.power(Cm2.diagonal(), 0.5)

# predicted data
zeval1 = 10 * np.arange(11) / 10
Go1 = np.array([np.ones(11), zeval1])
deval1 = Go1.transpose().dot(mest1)
Cdeval1 = Go1.transpose().dot(Cm1.transpose()).dot(Go1)
sdeval1 = np.power(Cdeval1.diagonal(), 0.5)
zeval2 = 10 * np.arange(11) / 10
Go2 = np.array([np.ones(11), zeval2])
deval2 = Go2.transpose().dot(mest2)
Cdeval2 = Go2.transpose().dot(Cm2.transpose()).dot(Go2)
sdeval2 = np.power(Cdeval2.diagonal(), 0.5)

# build figure
plt.subplots(1, 2, figsize=(15, 5))

# CASE 1: clumped data
plt.subplot(121)
plt.scatter(z1, dobs1, color='black')
plt.plot(zeval1, deval1, color='red')
plt.plot(zeval1, deval1+sdeval1, color='blue')
plt.plot(zeval1, deval1-sdeval1, color='blue')
plt.xlim(0,10)
plt.ylim(0,15)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Clumped Data')

# CASE 2: spread data
plt.subplot(122)
plt.scatter(z2, dobs2, color='black')
plt.plot(zeval2, deval2, color='red')
plt.plot(zeval2, deval2+sdeval2, color='blue')
plt.plot(zeval2, deval2-sdeval2, color='blue')
plt.xlim(0,10)
plt.ylim(0,15)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Spread Data')

# Dirichlet solution

# z-axis
z = np.arange(101) / 10

# build true model - mostly zero but a few spikes
mtrue = np.zeros(101)
mtrue[4] = 1
mtrue[9] = 1
mtrue[19] = 1
mtrue[49] = 1
mtrue[89] = 1

# experiment: exponential smoothing of model
c = np.arange(81) * 0.1
G = np.exp((-c).reshape(81,1) * z)
dtrue = G.dot(mtrue)
dobs = dtrue + np.random.normal(0, 1e-12, 81)

# minimum length solution
GMG = solve(G.dot(G.transpose()) + 1e-12 * np.eye(81), G).transpose()
mest = GMG.dot(dobs)
Rres = GMG.dot(G)

# build figure
plt.subplots(1, 2, figsize=(18, 5))

# true model parameters
plt.subplot(131)
plt.plot(z, mtrue, color='black')
plt.plot(z, mest, color='red')
plt.xlim(0,10)
plt.ylim(-0.2,1)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Dirichlet Approach')

# model resolution matrix
plt.subplot(132)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Rres)
plt.imshow(Rres, interpolation='nearest')
plt.plot((0,100), (0,100), color='black', ls='dashed')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,100)
plt.ylim(100,0)
plt.xlabel('j')
plt.ylabel('i')
plt.title('Model Resolution Matrix')

# Backus-Gilbert solution

# z-axis
z = np.arange(101) / 10

# build true model - mostly zero but a few spikes
mtrue = np.zeros(101)
mtrue[4] = 1
mtrue[9] = 1
mtrue[19] = 1
mtrue[49] = 1
mtrue[89] = 1

# experiment: exponential smoothing of model
c = np.arange(81) * 0.1
G = np.exp((-c).reshape(81,1) * z)
dtrue = G.dot(mtrue)
dobs = dtrue + np.random.normal(0, 1e-12, 81)

GMG = np.zeros((101,81))
u = G.dot(np.ones(101))

# construct Backus-Gilbert solution row-wise
for k in range(1, 101):
    S = G.dot(np.diag(np.power(np.arange(0, 101) - k, 2))).dot(G.transpose()) + 1e-6 * np.eye(81)
    uSinv = solve(S, u.transpose())
    GMG[k,] = uSinv / uSinv.dot(u)
    
# calculate model estimate and model resolution matrix
mest = GMG.dot(dobs)
Rres = GMG.dot(G)

# build figure
plt.subplots(1, 2, figsize=(18, 5))

# true model parameters
plt.subplot(131)
plt.plot(z, mtrue, color='black')
plt.plot(z, mest, color='red')
plt.xlim(0,10)
plt.ylim(-0.2,1)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Backus-Gilbert Approach')

# model resolution matrix
plt.subplot(132)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Rres)
plt.imshow(Rres, interpolation='nearest')
plt.plot((0,100), (0,100), color='black', ls='dashed')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,100)
plt.ylim(100,0)
plt.xlabel('j')
plt.ylabel('i')
plt.title('Model Resolution Matrix')

# COARSE DISCRETIZATION

# box
plt.subplots(1, 2, figsize=(10, 5))
ax1 = plt.subplot(121, aspect='equal')
ax1.add_patch(patches.Rectangle((0, 0), 2, 2, lw=3, ec='black', fill=False))
plt.xlim(-0.1,2.1)
plt.ylim(-0.1,2.1)

# circle
R = 0.9
x0 = 1
y0 = 1
th = 2 * math.pi * np.arange(101) / 100
x = x0 + R * np.sin(th)
y = y0 + R * np.cos(th)
plt.plot(x,y, linewidth=3)

# rays
x1 = np.random.uniform(0, 2, 21)
x2 = np.random.uniform(0, 2, 21)
y1 = np.random.uniform(0, 2, 21)
y2 = np.random.uniform(0, 2, 21)
for i in range(0,21):
    plt.plot((x1[i], x2[i]), (0, 2), color='gray')
    plt.plot((0, 2), (y1[i], y2[i]), color='gray')
    
# coarse discretization grid
xg = 0 + 2 * np.arange(7+1) / 7
yg = 0 + 2 * np.arange(7+1) / 7
for i in range(0,7+1):
    plt.plot((xg[0], xg[7]), (yg[i], yg[i]), color='black')
    plt.plot((xg[i], xg[i]), (yg[0], yg[7]), color='black')

plt.title('Coarse Discretization')
    
# FINE DISCRETIZATION

# box
ax2 = plt.subplot(122, aspect='equal')
ax2.add_patch(patches.Rectangle((0, 0), 2, 2, lw=3, ec='black', fill=False))
plt.xlim(-0.1,2.1)
plt.ylim(-0.1,2.1)

# circle
plt.plot(x,y, linewidth=3)

# rays
for i in range(0,21):
    plt.plot((x1[i], x2[i]), (0, 2), color='gray')
    plt.plot((0, 2), (y1[i], y2[i]), color='gray')
    
# fine discretization grid
xg = 0 + 2 * np.arange(21+1) / 21
yg = 0 + 2 * np.arange(21+1) / 21
for i in range(0,21+1):
    plt.plot((xg[0], xg[21]), (yg[i], yg[i]), color='black')
    plt.plot((xg[i], xg[i]), (yg[0], yg[21]), color='black')
    
plt.title('Fine Discretization')

# trade-off of resolution and variance

# z-axis
z = np.arange(101) / 10

# experiment: exponential smoothing of model
c = np.arange(81) * 0.1
G = np.exp((-c).reshape(81,1) * z)

# Part 1: Backus-Gilbert
alphavec = np.array([(0.999, 0.995, 0.99, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001)])
iahalf = np.where(alphavec == 0.5)[1][0]
spreadR1 = np.zeros(alphavec.shape[1])
sizeC1 = np.zeros(alphavec.shape[1])

for a in range(0, alphavec.shape[1]):
    # construct Backus-Gilbert solution row-wise
    GMG = np.zeros((101, 81))
    u = G.dot(np.ones(101))
    
    for k in range(0, 101):
        S = G.dot(np.diag(np.power(np.arange(0, 101) - k, 2))).dot(G.transpose())
        Sp = alphavec[0,a] * S + (1 - alphavec[0,a]) * np.eye(81)
        uSpinv = solve(Sp, u.transpose())
        GMG[k,] = uSpinv / uSpinv.dot(u)
        
    Cm1 = GMG.dot(GMG.transpose())
    R1 = GMG.dot(G)
    sizeC1[a] = np.sum(np.diag(Cm1))
    
    spreadR1[a] = np.sum(np.sum(
            np.multiply(np.square(np.arange(1,102).reshape(101,1) * np.ones(101) - np.ones(101).reshape(101,1) * np.arange(1,102)),
                       np.square(R1))))
    
    
# build figure
plt.subplots(1, 2, figsize=(10, 5))

# backus-gilbert approach
ax = plt.subplot(121)
plt.plot(np.log10(spreadR1), np.log10(sizeC1), color='black')
plt.xlim(2,3.5)
plt.ylim(-3,5)
plt.xlabel('log10 spread(R)')
plt.ylabel('log10 size(Model Covariance)')
plt.title('Backus-Gilbert Approach')
ax.text(2.3, 4, r'$\alpha=1$', fontsize=16)
ax.text(2.5, 1.25, r'$\alpha=\frac{1}{2}$', fontsize=16)
ax.text(3.1, -2, r'$\alpha=0$', fontsize=16)

# Part 2: Damped minimum length

alphavec = np.array([(0.9999, 0.999, 0.995, 0.99, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01)])
iahalf = np.where(alphavec == 0.5)[1][0]
spreadR2 = np.zeros(alphavec.shape[1])
sizeC2 = np.zeros(alphavec.shape[1])

for a in range(0, alphavec.shape[1]):
    GMG = solve(alphavec[0,a] * G.dot(G.transpose()) + (1 - alphavec[0,a]) * np.eye(81), alphavec[0,a] * G)
    
    Cm2 = GMG.dot(GMG.transpose())
    R2 = GMG.transpose().dot(G)
    sizeC2[a] = np.sum(np.diag(Cm2))
    spreadR2[a] = np.sum(np.square(R2 - np.eye(101)))

# damped minimum length
ax = plt.subplot(122)
plt.plot(spreadR2, np.log10(sizeC2), color='black')
plt.xlim(91,101)
plt.ylim(-3,5)
plt.xlabel('spread (R)')
plt.ylabel('log10 size (Model Covariance)')
plt.title('Damped Minimum Length Approach')
ax.text(93, 3.75, r'$\alpha=1$', fontsize=16)
ax.text(97, -0.5, r'$\alpha=\frac{1}{2}$', fontsize=16)
ax.text(99, -2.25, r'$\alpha=0$', fontsize=16)

# Checkerboard resolution test

# inverse problem is an acoustic tomography problem, where observations are made along rows and columns

# grid of unknowns is Lx by Ly
Lx = 20
Ly = 20
M = Lx * Ly

# observations along rows and columns
N = Lx + Ly

# build backward index tables for convenience
ixofj = np.zeros(M)
iyofj = np.zeros(M)
for ix in range(0, Lx):
    for iy in range(0, Ly):
        # map model param at (ix, iy) into scalar index j
        j = ix * Ly + iy 
        ixofj[j] = ix
        iyofj[j] = iy
        
G = np.zeros((N,M))

# observations across rows
for ix in range(0, Lx):
    for iy in range(0, Ly):
        j = ix * Ly + iy
        G[ix, j] = 1
        
# observations across columns
for iy in range(0, Ly):
    for ix in range(0, Lx):
        j = ix * Ly + iy
        G[iy + Lx, j] = 1
        
# model parameter vector mk for crude checkerboard
mk = np.zeros(M)
for ix in range(0, Lx, 4):
    for iy in range(0, Ly, 4):
        mk[ix * Ly + iy] = 1
        
# predicted data
dk = G.dot(mk)

# solve inverse problem and interpret the result as a row of the resolution matrix
GMG = solve(G.dot(G.transpose()), G)
rk = GMG.transpose().dot(dk)

# reorganize to 2D physical model space
Rk = np.zeros((Lx, Ly))
checkerboard = np.zeros((Lx, Ly))

for i in range(0,M):
    Rk[ixofj[i], iyofj[i]] = rk[i]
    checkerboard[ixofj[i], iyofj[i]] = mk[i]
    
# build figure
plt.subplots(1, 2, figsize=(12, 10))

# true checkerboard
plt.subplot(221)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(checkerboard)
plt.imshow(checkerboard, interpolation='nearest')
plt.grid(True)
plt.title('True Checkerboard')

# reconstructed checkerboard
plt.subplot(222)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Rk)
plt.imshow(Rk, interpolation='nearest')
plt.grid(True)
plt.colorbar(m)
plt.title('Reconstructed Checkerboard')

