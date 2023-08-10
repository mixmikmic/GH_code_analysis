# least squares fit of synthetic data

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# simulate independent variable
z = np.sort(np.random.uniform(0, 10, 30))

# simulate dependent/observed variable
# d = a + b*z + noise
dobs = 2.0 + 1.0 * z + np.random.normal(0,1.0,30)

# least squares fit
M = 2
G = np.array([np.ones(30), z])
mest = inv(G.dot(np.transpose(G))).dot(G.reshape(2,30).dot(dobs)).reshape(2,1)

# predicted data & error
dpre = np.transpose(G).dot(mest)
e = dobs - dpre

# maximum error
iemax = abs(e).argmax()
emax = np.max(abs(e))

# build figure
plt.subplots(1,2, figsize=(10,5))

# PLOT 1: Simulated data and least squares solution
plt.subplot(121)

# simulated data
plt.scatter(z, dobs, color="black")

# least squares solution
plt.plot(z, dpre, color="blue")

# graphing params
plt.xlim(0,10)
plt.ylim(0,15)
plt.title('Least Squares Fitting')
plt.xlabel('z')
plt.ylabel('d')

# PLOT 2: Least squares solution demonstrating error as difference between observed and predicted datum
ax = plt.subplot(122)

# least squares solution
plt.plot(z, dpre, color="blue")

# annotating lines
plt.plot([z[iemax], z[iemax]], [dpre[iemax], dobs[iemax]], color="red", lw=5)
plt.plot([z[iemax], z[iemax]], [0,dpre[iemax]], color="black", ls='dashed')
plt.plot([0, z[iemax]], [dpre[iemax], dpre[iemax]], color="black", ls='dashed')
plt.plot([0, z[iemax]], [dobs[iemax], dobs[iemax]], color="black", ls='dashed')
plt.scatter([z[iemax]], [dobs[iemax]], color="blue", s=100)
ax.text(z[iemax]-0.75, (dobs[iemax]+dpre[iemax])/2, '$e_i$', fontsize=12)
ax.text(2, dobs[iemax]+0.5, '$d_i^{obs}$', fontsize=12)
ax.text(2, dpre[iemax]-1, '$d_i^{pre}$', fontsize=12)
ax.text(z[iemax]-0.75, 2, '$z_i$', fontsize=12)

# graphing params
plt.xlim(0,10)
plt.ylim(0,15)
plt.title('Error Definition')
plt.xlabel('z')
plt.ylabel('d')

# HYPOTHETICAL PREDICTION ERROR

# x-axis data
z = np.arange(31) / 3

# randomly sampled "errors" from uniform distribution
e = np.random.uniform(-1, 1, 31)

# calculate different norms
e1 = abs(e)
E1 = sum(e1)
e2 = np.power(abs(e), 2)
E2 = sum(e2)**0.5
e10 = np.power(abs(e), 10)
E10 = sum(e10)**0.1

# output error summary
print("Norms\n", "E1:", E1, "E2:", E2, "E10:", E10)

# build figure
plt.subplots(4, 1, figsize=(15, 10))
plt.tight_layout(h_pad=3)

# plot errors
plt.subplot(411)
plt.bar(z-(z[1]-z[0]), e, width=z[1]-z[0])
plt.xlim(0,10)
plt.ylim(-1,1)
plt.xlabel('z')
plt.ylabel('$e$')
plt.title('Hypothetical Prediction Errors')

# plot absolute errors
plt.subplot(412)
plt.bar(z-(z[1]-z[0]), e1, width=z[1]-z[0])
plt.xlim(0,10)
plt.ylim(-1,1)
plt.xlabel('z')
plt.ylabel('$|e|$')

# plot squared errors
plt.subplot(413)
plt.bar(z-(z[1]-z[0]), e2, width=z[1]-z[0])
plt.xlim(0,10)
plt.ylim(-1,1)
plt.xlabel('z')
plt.ylabel('$|e|^2$')

# plot errors to the tenth order
plt.subplot(414)
plt.bar(z-(z[1]-z[0]), e10, width=z[1]-z[0])
plt.xlim(0,10)
plt.ylim(-1,1)
plt.xlabel('z')
plt.ylabel('$|e|^{10}$')

# STRAIGHT LINE FITS UNDER DIFFERENT NORMS

# simulated x-axis data
z = np.sort(np.random.uniform(0, 10, 30))

# simulated observed data
dobs = 2.0 + 1.0 * z + np.random.normal(0, 0.5, 30)

# one terrible outlier
dobs[dobs.shape[0]-1] = 1

# populate a grid with errors
E1 = np.zeros((101, 101))
E2 = np.zeros((101, 101))
Einf = np.zeros((101, 101))
for i in range(0, 101):
    for j in range(0, 101):
        # predicted data
        a0 = 0.04 * (i-1)
        b0 = 0.04 * (j-1)
        dpre = a0 + b0 * z
        
        # calculate errors
        e = dobs - dpre
        
        # calculate norm matrix elements
        abse = abs(e)
        E1[i, j] = np.sum(abse)
        E2[i, j] = np.sum(np.power(abse, 2))
        Einf[i, j] = np.sum(np.power(abse, 20))   # cheating; using large but finite power

# define predicted data for L1 norm
i1 = np.unravel_index(E1.argmin(), E1.shape)
dpre1 = 0.04*(i1[0]) + 0.04*(i1[1]) * z

# define predicted data for L2 norm
i2 = np.unravel_index(E2.argmin(), E2.shape)
dpre2 = 0.04*(i2[0]) + 0.04*(i2[1]) * z

# define predicted data for Linf norm
iinf = np.unravel_index(Einf.argmin(), Einf.shape)
dpreinf = 0.04*(iinf[0]) + 0.04*(iinf[1]) * z

# build figure
plt.subplots(1, 1, figsize=(7, 5))
ax = plt.subplot(111)

# plot observed data
plt.scatter(z, dobs, color='black')

# plot L1, L2, and Linf norm's
plt.plot(z, dpre1, color='red')
plt.plot(z, dpre2, color='green')
plt.plot(z, dpreinf, color='blue')

# annotations
ax.text(9, dpre1[29]+0.25, '$L_1$', fontsize=12)
ax.text(9, dpre2[29]-1.5, '$L_2$', fontsize=12)
ax.text(9, dpreinf[29]+0.25, '$L_\infty$', fontsize=12)
ax.annotate('Outlier', xy=(z[29], dobs[29]), xytext=(7, 3), arrowprops=dict(facecolor='black', shrink=0.1),)

# plot params
plt.xlim(0,10)
plt.ylim(0,15)
plt.xlabel('z')
plt.ylabel('d')
plt.title('Straight line fits using $L_1$, $L_2$, and $L_\infty$ norms')

# LONG VS SHORT-TAILED PROBABILITY DENSITY FUNCTIONS

import math

# x-axis data
d = np.arange(101) / 10

# short-tailed probability density function (Normal PDF)
dbar = 5
sd = 1.0
d2 = np.power(d-dbar, 2)
p1 = np.exp(-0.5 * d2 / sd**2) / (math.sqrt(2*math.pi)*sd)
A1 = 0.1 * sum(p1)

# long-tailed distribution (Cauchy-Lorentz distribution)
g = 1
p2 = 1 / (math.pi * g * (1 + d2 / (g**2)))
A2 = 0.1 * sum(p2)

# check on areas
print('Check on areas\n', 'True:', 1, 'Estimate 1:', A1, 'Estimate 2:', A2)

# build figure
plt.subplots(1, 2, figsize=(12, 5))
plt.tight_layout(w_pad=5)
ax = plt.subplot(121)

plt.plot(d, p2, color='blue')
plt.xlim(0,10)
plt.ylim(0,0.5)
plt.xlabel('d')
plt.ylabel('p(d)')
plt.title('Long-tailed PDF')

ax = plt.subplot(122)

plt.plot(d, p1, color='red')
plt.xlim(0,10)
plt.ylim(0,0.5)
plt.xlabel('d')
plt.ylabel('p(d)')
plt.title('Short-tailed PDF')

# STRAIGHT LINE PROBLEM: Least squares fit to synthetic data

# x-axis synthetic data
z = np.sort(np.random.uniform(0, 10, 30))

# observed synthetic data (d = a + bz + noise)
dobs = 2.0 + 1.0 * z + np.random.normal(0, 1.0, 30)

# data kernel
M = 2
G = np.array([np.ones(30), z])

# standard matrix solution
mest1 = inv(G.dot(np.transpose(G))).dot(G.dot(dobs)).reshape(2,1)
dpre1 = np.transpose(G).dot(mest1)

# LARGER PROBLEMS - Use biconjugate gradient algorithm

from scipy.sparse.linalg import bicg

# biconjugate gradient solution (THIS PROBABLY ISN'T RIGHT ...)
mest2 = bicg(G.dot(np.transpose(G)), G.dot(dobs), tol=1e-06, maxiter=90)[0]
dpre2 = np.transpose(G).dot(mest2)

# build figure
plt.subplots(1,1, figsize=(7,5))

# PLOT 1: Simulated data and least squares solution
plt.subplot(111)

# simulated data
plt.scatter(z, dobs, color="black")

# least squares solution and biconjugate gradient solution
plt.plot(z, dpre1, color="blue")
plt.plot(z, dpre2, color='red', ls='dashed', lw=2)

# graphing params
plt.xlim(0,10)
plt.ylim(0,15)
plt.title('Least Squares Fitting')
plt.xlabel('z')
plt.ylabel('d')

import pandas as pd

# read in data and print first few rows
df = pd.read_csv("../data/planetary.txt", delim_whitespace=True, header=None, names=["Radius", "Period"])
print(df.head())

# load radius and period into numpy arrays
#    divide by 10**9 and 10**3 respectively - computers don't like really big numbers (especially when applying powers)
radius = df.Radius.values / 10**9
period = df.Period.values / 10**3

# take radius**3 to be the observation, period to be auxiliary variable
dobs = np.power(radius, 3)
z = period

# build system matrix G
G = np.array([np.ones(z.shape[0]), z, np.power(z, 2)])
mest = inv(G.dot(np.transpose(G))).dot(G.dot(dobs))
print('mest=\n', mest)

# calculate predicted data and error
dpre = np.transpose(G).dot(mest)
e = dobs - dpre

# build smooth parabola for display, lot's of z's!
zeval = np.arange(0, 251) / 2.5
deval = mest[0] + mest[1] * zeval + mest[2] * np.power(zeval, 2)

plt.subplots(2, 1, figsize=(10, 5))

plt.subplot(211)
plt.scatter(z, dobs, color='blue')
plt.plot(zeval, deval, color='black')
plt.xlim(0,100)
plt.ylim(0,250)
plt.xlabel(r'$Period, kdays$')
plt.ylabel(r'$Radius^3 \: (Tm^3)$')
plt.title('Kepler\'s Third Law')

plt.subplot(212)
plt.scatter(z, e, color='red')
plt.plot(zeval, np.zeros(zeval.shape[0]), color='black')
plt.xlim(0,100)
plt.ylim(-0.5, 0.5)
plt.xlabel(r'$Period, kdays$')
plt.ylabel(r'$Error \: (Tm^3)$')

from mpl_toolkits.mplot3d import Axes3D

# read Kurile subduction zone data
df = pd.read_csv("../data/kurile_eqs.txt", delim_whitespace=True, header=None, names=["lat", "lon", "depth"])
print(df.head())

# convert to array and fix units to kilometers
x = 111.12 * np.cos((math.pi/180) * np.mean(df.lat.values)) * (df.lon.values - np.min(df.lon.values))
y = 111.12 * (df.lat.values - np.min(df.lat.values))
z = -1 * df.depth.values

# build figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot data
ax.scatter(-x, -y, z)

# graph params
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Depth (km)')
ax.set_xlim3d(-1200, 0)
ax.set_ylim3d(-1200, 0)
ax.set_zlim3d(-700, 0)

from matplotlib.colors import Normalize
from matplotlib import cm

# setup and solve the inverse problem
G = np.array([np.ones(z.shape[0]), x, y])
mest = inv(G.dot(np.transpose(G))).dot(G.dot(z))
dpre = np.transpose(G).dot(mest)

# setup and calculate values for mesh surface
xx = 1200 * np.arange(0, 31) / 30
yy = 1200 * np.arange(0, 31) / 30
X, Y = np.meshgrid(xx, yy)
Z = mest[0] + mest[1] * X + mest[2] * Y

# build figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# choose any colormap e.g. cm.jet, cm.coolwarm, etc.
color_map = cm.RdYlGn # reverse the colormap: cm.RdYlGn_r
scalarMap = cm.ScalarMappable(norm=Normalize(vmin=-700, vmax=0), cmap=color_map)
C = scalarMap.to_rgba(Z)

# plot data and model
ax.scatter(-x, -y, z)
ax.plot_surface(-X, -Y, Z, rstride=1, cstride=1, facecolors=C, antialiased=True, alpha=0.3)

# graph params
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Depth (km)')
ax.set_xlim3d(-1200, 0)
ax.set_ylim3d(-1200, 0)
ax.set_zlim3d(-700, 0)

# single data point
xd = np.array([2])
yd = np.array([2])

# x-axis
x = np.arange(1,4+0.1,0.1)

# first solution: y=x
y1 = x

# second solution: y=2x-2
y2 = 2 * x - 2

# third solution: y=0.5x+1
y3 = 0.5 * x + 1

# build figure
plt.subplots(1,1, figsize=(10,8))
ax = plt.subplot(111)

# single data point
plt.scatter(xd, yd, color="black")

# least squares solution and biconjugate gradient solution
plt.plot(x, y1, color="black")
plt.plot(x, y2, color='black')
plt.plot(x, y3, color="black")

ax.text(4.2, 4.2, '?', fontsize=12)
ax.text(4.2, 6.4, '?', fontsize=12)
ax.text(4.2, 3.1, '?', fontsize=12)

# graphing params
plt.xlim(0,5)
plt.ylim(-1,7)
plt.title('$\infty$ different lines can pass through a single point with zero error')
plt.xlabel('x')
plt.ylabel('y')
ttl = ax.title
ttl.set_position([.5, 1.05])

from scipy.sparse import csr_matrix

# sample data from a sinusoid in an auxiliary variable z
M = 101
Dz = 1.0
z = np.arange(0,M)
zmax = np.max(z)
mtrue = np.sin(3 * math.pi * z / zmax)

# let us suppose that we observe the sine wave at the following z indices
index = np.array([(1, 9, 15, 17, 37, 49, 61, 73, 85, 91, M-1)])
N = index.shape[1]
zobs = z[index]
dobs = np.sin(3 * math.pi * zobs/zmax) + np.random.normal(0, 0.05, N)

# the N data equations are just m=dobs.  The only trick is lining up the corresponding elements of m and dobs,
# since they are not the same length
F = np.zeros((N+M, M))
f = np.zeros(N+M)
for i in range(0, N-1):
    F[i, index[0][i]] = 1
    f[i] = dobs[0][i]

# now implement the 2nd derivative smoothness constraints of all interior m's
epsilon = 1.0
rDz = 1/Dz
rDz2 = 1 / Dz**2
for i in range(0, M-2):
    F[i+N, i] = epsilon * rDz2
    F[i+N, i+1] = -2 * epsilon * rDz2
    F[i+N, i+2] = epsilon * rDz2
    f[i+N] = 0

# now implement 1st derivative flatness constraints for m's at edge
F[N+M-2, 1] = -epsilon * rDz
F[N+M-2, 2] = epsilon * rDz
F[N+M-1, M-2] = -epsilon * rDz
F[N+M-1, M-1] = epsilon * rDz
f[N+M-1] = 0
    
# now convert to a sparse matrix data type, and solve using biconjugate gradient
F = csr_matrix(F)
mest = bicg(F.transpose().dot(F), F.transpose().dot(f), tol=1e-06, maxiter=3*M)[0]

# build figure
plt.subplots(1,1, figsize=(10,8))
ax = plt.subplot(111)

# single data point
plt.scatter(zobs, dobs, color="black")

# least squares solution and biconjugate gradient solution
plt.plot(z, mtrue, color="black")
plt.plot(z, mest, color="blue")

# graphing params
plt.xlim(0,100)
plt.ylim(-1,1)
plt.xlabel('z')
plt.ylabel('m(z)')

from numpy.linalg import inv

# constrained least squares to fit synthetic data

# z-axis
z = np.sort(np.random.uniform(0, 10, 30))
dobs = 2 + z + np.random.normal(0, 0.5, 30)

# constraining point
zp = 8
dp = 6

# constrained least squares fit
G = np.array([np.ones(30), z])
GTG = G.dot(np.transpose(G))
GTd = G.dot(dobs)
F = np.array([1, zp])

# build system matrix
A = np.zeros((3,3))
for i in range(0,2):
    A[i,2] = F[i]
    for j in range(0,2):
        A[i,j] = GTG[i,j]
        A[2,j] = F[j]

# build right vector
b = np.zeros(3)
for i in range(0, 2):
    b[i] = GTd[i]
b[2] = dp

# solve for model estimate and predict data
mest = inv(A).dot(b)
dpre = np.transpose(G).dot(mest[0:2])

# build figure
plt.subplots(1,1, figsize=(7,5))
ax = plt.subplot(111)

# single constraining data point
plt.scatter(zp, dp, color="blue", s=100)

# observed data points
plt.scatter(z, dobs, color="red")

# least squares solution and biconjugate gradient solution
plt.plot(z, dpre, color="black")

# graph annotation
ax.text(2, dp+1, r'$d_p$', fontsize=12)
ax.text(zp-0.5, 2, r'$z_p$', fontsize=12)
plt.plot([zp, zp], [0,dp], color="black", ls='dashed')
plt.plot([0, zp], [dp,dp], color="black", ls='dashed')
ax.annotate('Constraint', xy=(zp, dp), xytext=(4, 2), arrowprops=dict(facecolor='black', shrink=0.1),)

# graphing params
plt.xlim(0,10)
plt.ylim(0,15)
plt.xlabel('z')
plt.ylabel('d')

from scipy.linalg import toeplitz

# two hypothetical experiments to measure the weight m_i of each of 100 bricks
# EXPERIMENT 1: the bricks are accumulated on a scale so that data observation i is the sum of the weight of the first i bricks
# EXPERIMENT 2: the first brick is weighed, and then subsequent pairs of bricks are weighed (1st, 1st & 2nd, 2nd & 3rd, ...)

# build TRUE model
z = np.arange(100)+1
mtrue = np.ones(100)

# build system matrices (experiment 1 and 2)
G1 = toeplitz(np.ones(100), np.zeros(100))
G2 = toeplitz(np.array([1, 1, *np.zeros(100-2).tolist()]), np.array([1, *np.zeros(100-1).tolist()]))

# TRUE data observations (experiments 1 and 2)
d1true = G1.dot(mtrue)
d2true = G2.dot(mtrue)

# add Gaussian noise to the data (experiments 1 and 2)
d1obs = d1true + np.random.normal(0, 0.1, 100)
d2obs = d2true + np.random.normal(0, 0.1, 100)

# solve for model parameters (both experiments)
m1est = bicg(np.transpose(G1).dot(G1), np.transpose(G1).dot(d1obs))[0]
m2est = bicg(np.transpose(G2).dot(G2), np.transpose(G2).dot(d2obs))[0]

# calculate covariance matrices and corresponding error for each model parameter (experiments 1 adn 2)
C1 = 0.1**2 * inv(G1.transpose().dot(G1))
sm1 = np.power(C1.diagonal(), 0.5)
C2 = 0.1**2 * inv(G2.transpose().dot(G2))
sm2 = np.power(C2.diagonal(), 0.5)

# build figure
plt.subplots(2, 1, figsize=(10, 5))

plt.subplot(211)
plt.plot(z, mtrue, color='black')
plt.plot(z, m1est, color='red')
plt.plot(z, m2est, color='blue')
plt.xlim(0,100)
plt.ylim(-2.5, 2.5)
plt.xlabel('i')
plt.ylabel('$m_i^{\text{est}}$')

plt.subplot(212)
plt.plot(z, sm1, color='red')
plt.plot(z, sm2, color='blue')
plt.xlim(0,100)
plt.ylim(0, 1)
plt.xlabel('i')
plt.ylabel('$\sigma_{m_i}$')

import matplotlib.cm as cm

# examine error surface of intercept and slope in a straight line fit to synthetic data

# z-axis
z = np.sort(np.random.uniform(-5,5,30))

# PART ONE: data evenly spread along interval
# d = a + bz + noise
dobs = 2 + 1 * z + np.random.normal(0, 0.51, 30)

# grid
a = np.arange(100)*4/100
b = np.arange(100)*4/100

# populate grid with errors
EA = np.zeros((100, 100))
for i in range(0,100):
    for j in range(0,100):
        ao = 4/100 * (i-1)
        bo = 4/100 * (j-1)
        dpre = ao + bo * z
        e = dobs - dpre
        EA[i,j] = e.transpose().dot(e)

# find minimum error
EAmin = np.min(EA)
print('EA min:', np.min(EA))
EAminRC = np.unravel_index(np.argmin(EA), EA.shape)
a1 = 4/100 * (EAminRC[0]-1)
b1 = 4/100 * (EAminRC[1]-1)
dpre = a1 + b1 * z

# calculate covariance
G = np.array([np.ones(30), z])
C1 = 0.51**2 * inv(G.transpose().dot(G))

# curvature of error surface
j=1
d2Eda2 = (EA[EAminRC[0]+j, EAminRC[1]] - 2*EA[EAminRC[0],EAminRC[1]] + EA[EAminRC[0]-j,EAminRC[1]]) / ((j * 4/100)**2)
d2Edb2 = (EA[EAminRC[0], EAminRC[1]+j] - 2*EA[EAminRC[0],EAminRC[1]] + EA[EAminRC[0],EAminRC[1]-j]) / ((j * 4/100)**2)
d2Edadb = (EA[EAminRC[0]+j, EAminRC[1]+j] - 
           EA[EAminRC[0]+j, EAminRC[1]-j] - 
           EA[EAminRC[0]-j, EAminRC[1]+j] + 
           EA[EAminRC[0]-j, EAminRC[1]-j]) / (4*j*4/100*j*4/100)
DA = np.zeros((2,2))
DA[0,0] = d2Eda2
DA[0,1] = d2Edadb
DA[1,0] = d2Edadb
DA[1,1] = d2Edb2
C2 = 0.51**2 * inv(DA/2)
deltaE = (np.max(EA) - np.min(EA)) /100

# build figure
plt.subplots(2, 2, figsize=(10, 10))

# PART ONE least squares fitting of straight line
plt.subplot(221)
plt.scatter(z, dobs, color='red')
plt.plot(z, dpre, color='blue')
plt.xlim(-5,5)
plt.ylim(-10,10)
plt.xlabel('z')
plt.ylabel('d')

# PART ONE error surface
plt.subplot(223)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(EA)
plt.imshow(EA, interpolation='nearest', extent=(0,4,4,0))
plt.contour(CS, hold='on', levels=[np.min(EA)+deltaE], colors='white', extent=(0,4,4,0))
plt.scatter(b1, a1, color='white')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,4)
plt.ylim(4,0)
plt.xlabel('$m_2$')
plt.ylabel('$m_1$')

# PART TWO: data bunched up at end of interval

# z-axis
z = np.sort(np.random.uniform(-5+1*10/2, 5, 30))

# d = a + bz + noise
dobs = 2 + 1 * z + np.random.normal(0, 0.51, 30)

# grid
a = np.arange(100)*4/100
b = np.arange(100)*4/100

# populate grid with errors
EB = np.zeros((100, 100))
for i in range(0,100):
    for j in range(0,100):
        ao = 4/100 * (i-1)
        bo = 4/100 * (j-1)
        dpre = ao + bo * z
        e = dobs - dpre
        EB[i,j] = e.transpose().dot(e)
        
# find minimum error
EBmin = np.min(EB)
print('EB min:', np.min(EB))
EBminRC = np.unravel_index(np.argmin(EB), EB.shape)
a1 = 4/100 * (EBminRC[0]-1)
b1 = 4/100 * (EBminRC[1]-1)
dpre = a1 + b1 * z

# calculate covariance
G = np.array([np.ones(30), z])
C1 = 0.51**2 * inv(G.transpose().dot(G))

# curvature of error surface
j=1
d2Eda2 = (EB[EBminRC[0]+j, EBminRC[1]] - 2*EB[EBminRC[0],EBminRC[1]] + EB[EBminRC[0]-j,EBminRC[1]]) / ((j * 4/100)**2)
d2Edb2 = (EB[EBminRC[0], EBminRC[1]+j] - 2*EB[EBminRC[0],EBminRC[1]] + EB[EBminRC[0],EBminRC[1]-j]) / ((j * 4/100)**2)
d2Edadb = (EB[EBminRC[0]+j, EBminRC[1]+j] - 
           EB[EBminRC[0]+j, EBminRC[1]-j] - 
           EB[EBminRC[0]-j, EBminRC[1]+j] + 
           EB[EBminRC[0]-j, EBminRC[1]-j]) / (4*j*4/100*j*4/100)
DB = np.zeros((2,2))
DB[0,0] = d2Eda2
DB[0,1] = d2Edadb
DB[1,0] = d2Edadb
DB[1,1] = d2Edb2
C2 = 0.51**2 * inv(DB/2)
deltaE = (np.max(EB) - np.min(EB)) /100

# PART ONE least squares fitting of straight line
plt.subplot(222)
plt.scatter(z, dobs, color='red')
plt.plot(z, dpre, color='blue')
plt.xlim(-5,5)
plt.ylim(-10,10)
plt.xlabel('z')
plt.ylabel('d')

# PART ONE error surface
plt.subplot(224)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(EB)
plt.imshow(EB, interpolation='nearest', extent=(0,4,4,0))
plt.contour(CS, hold='on', levels=[np.min(EB)+deltaE], colors='white', extent=(0,4,4,0))
plt.scatter(b1, a1, color='white')
plt.grid(True)
plt.colorbar(m)
plt.xlim(0,4)
plt.ylim(4,0)
plt.xlabel('$m_2$')
plt.ylabel('$m_1$')

