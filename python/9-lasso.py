# Plot f(\beta) and \hat{\beta} under 0 < y < lambda and y > lambda respectively.

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
#import seaborn as sns

import numpy as np

def f(y, beta, lam=1):
    return 0.5 * (y-beta)**2 + lam*np.abs(beta)

#  0 < y < lambda 
y = 0.5
beta = np.linspace(-1, 1.0, 21)
f1 = f(y, beta)

#  y > lambda
y2 = 2
beta2 = np.linspace(-0.5, 1.5, 21)
f2 = f(y2, beta2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.ylim((0, 2.5))
plt.xlabel('beta')
plt.ylabel('f(beta)')
plt.plot(beta, f1)
plt.title('y = 0.5, lambda = 1')
plt.annotate('optimal: 0', xy=(0, 0.125), xytext=(0.1, 0.1))
plt.plot([0], [0.125], 'o', color='red')

plt.subplot(1, 2, 2)
plt.ylim((1, 3.6))
plt.xlabel('beta')
plt.ylabel('f(beta)')
plt.plot(beta2, f2)
plt.title('y = 2, lambda = 1')
plt.annotate('optimal: y-lambda', xy=(1, 1.5), xytext=(0.5, 1.7))
plt.plot([1], [1.5], 'o', color='red')

# Plot beta hat.

def beta_hat(y, lam=1):
    return np.sign(y) * np.maximum(0, np.abs(y)-lam)

y = np.linspace(-2, 2, 41)
b = beta_hat(y)

plt.figure()
plt.grid(True)
plt.title('Soft thresholding. lambda = 1')
plt.ylabel('beta')
plt.xlabel('y')
plt.plot(y, b)

lam = 1.8
y = 1.5
beta = np.linspace(-1, 1, 21)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(beta, y*beta, label='y*beta')
plt.plot(beta, lam*np.abs(beta), label='lambda*|beta|', color='red')
plt.xlabel('beta')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(beta, y*beta, label='y*beta')
plt.plot(beta, lam*(beta**2), label='lambda*beta^2', color='red')
plt.annotate('f(b) < f(0)\nbeta would be the solution', xy=(0.3, 0.5), xytext=(-0.5, 1),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend(loc='best')
plt.xlabel('beta')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(beta, y*beta, label='y*beta')
plt.plot(beta, lam*np.power(np.abs(beta),0.5), label='lambda*|beta|^(1/2)', color='red')
plt.annotate('Locally good.\nBut f(b) < f(0) later.', xy=(1.0, 1.6), xytext=(-0.2, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.legend(loc='best')
plt.xlabel('beta')
plt.grid(True)

from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.figure(figsize=(12,4))

ax = plt.subplot(1,3,1)
path = Path([(-2, 0), (0, 2), (2, 0), (0, -2)])
patch = PathPatch(path, label='feasible region', alpha=0.2)
ax.add_patch(patch)
plt.axis('equal')
plt.xlim([-3, 3])
plt.title('L1: $|beta|_{l_1}\leq t$')

plt.subplot(1,3,2)
an = np.linspace(0, 2*np.pi, 100)
plt.fill_between(2*np.cos(an), 0, 2*np.sin(an), alpha=0.2)
plt.axis('equal')
plt.xlim([-3, 3])
plt.title('L2: $|beta|_{l_2}^2\leq t$')

plt.subplot(1,3,3)
plt.fill_between(np.linspace(-2, 2),
                 np.power(np.abs(np.linspace(-2, 2)), 0.5)-1.4,
                 1.4-np.power(np.abs(np.linspace(-2, 2)), 0.5), alpha=0.2)
plt.xlim([-3, 3])
plt.ylim([-2, 2])
plt.title('L1/2: $|beta|_{l_{1/2}} \leq t$')

n = 50
p = 200   # total dimensions
s = 10    # first 10 components are non-zero.
T = 10
lambda_all = np.arange(1200, 0, -1)
L = len(lambda_all)

np.random.seed(1)
# generate X and response Y (using s components)
X = np.random.standard_normal(size=(n,p))
beta_true = np.zeros(p)
beta_true[0:s] = np.arange(1, s+1)
Y = X.dot(beta_true) + np.random.standard_normal(n)

# we want to save the solution path in beta_all
beta = np.zeros(p)
beta_all = np.zeros((p, L))

R = Y
ss = np.zeros(p)
for j in xrange(p):
    ss[j] = sum(X[:, j]**2)
    
# different lambda
for l in xrange(L):
    lam = lambda_all[l]
    for t in xrange(T):
        # coordinate descent with a systematic scan.
        for j in xrange(p):
            # Deterimine an update amount followed by a soft-threasholding.
            db = np.sum(R * X[:, j]) / ss[j]
            beta_j = beta[j] + db
            beta_j = np.sign(beta_j) * max(0, abs(beta_j) - lam / ss[j])
            # update the coefficient and residue.
            delta = beta_j - beta[j]
            R = R - X[:, j] * delta
            beta[j] = beta_j
    beta_all[:, l] = beta

# plot the solution path.
norm = np.sum(np.abs(beta_all.T), 1)
plt.xlabel('|beta|')
plt.ylabel('beta')
plt.title('LASSO solution path')
plt.plot(norm, beta_all.T);

T = 1200
epsilon = .0001
beta = np.zeros(p)
db = np.zeros(p)
beta_all = np.zeros((p, T))

R = Y
for t in xrange(T):
    for j in xrange(p):
        db[j] = np.sum(R*X[:, j])
    j = np.argmax(np.abs(db)) 
    delta = db[j] * epsilon   # each time, only update a tiny step.
    beta[j] = beta[j] + delta
    R = R - X[:, j] * delta
    beta_all[:, t] = beta

norm = np.sum(np.abs(beta_all.T), 1)
plt.xlabel('|beta|')
plt.ylabel('beta')
plt.title('LASSO solution path')
plt.plot(norm, beta_all.T);



