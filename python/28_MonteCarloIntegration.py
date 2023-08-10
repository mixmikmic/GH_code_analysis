import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12,8))
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
f = np.exp(-X**2 - Y**2 - X**2*Y**2)
surf = ax.plot_surface(X, Y, f, rstride=2, cstride=2, cmap='bwr', linewidth=0, antialiased=True)

ax.set_xlabel('x',labelpad=14)
ax.set_ylabel('y',labelpad=14)
ax.set_zlabel('z',labelpad=14)

[t.label.set_fontsize(14) for caxis in [ax.xaxis, ax.yaxis, ax.zaxis] for t in caxis.get_major_ticks()];

def f(x,y):
    return np.exp(-x**2 - y**2 - x**2*y**2)

# Number of MC points
M = 10000

# initialize the integral
I = 0

for i in range(M):
    x = -0.5 + np.random.random()
    y = -0.5 + np.random.random()
    I += f(x,y)
    
I /= M

print('I = ',I)

def Monte_Carlo_integration(f,M):
    r = -0.5 + np.random.random([M,2])
    return np.average(f(r[:,0],r[:,1]))

exact = 0.846008

# we consider different numbers of MC points and many trials
M = np.array(range(1,100,1))
trials = 1000

I = np.zeros([trials,M.size])

# for each trial and number of points, compute the itegral
for trial in range(trials):
    for j,cM in enumerate(M):
        I[trial,j] = Monte_Carlo_integration(f,cM)

# compute the average and standard error in the integral
aveI = np.average(I,axis=0)
errI = np.std(I,axis=0)/np.sqrt(trials)

# the error in the integral
error = np.abs(aveI - exact)

from scipy.optimize import curve_fit

def fit_func(x,*a):
    return a[0]/x**a[1]

# fit to extract the form of the error
a1,a1_cov = curve_fit(fit_func,M,error,sigma=errI,p0=(1,1))
plt.plot(M,fit_func(M,*a1), color=colors[1], zorder=0, label=r'$M^{-%2.1f}$'%a1[1])

# compare with the calculated value
plt.errorbar(M,error,yerr=errI, marker='o', capsize=0, elinewidth=1, mec=colors[0], 
             mfc=colors[0], ms=6, linestyle='None', ecolor=colors[0], label='Monte Carlo Integration')

plt.ylabel('Error')
plt.xlabel('Number MC Points')
plt.legend(loc='upper right')



