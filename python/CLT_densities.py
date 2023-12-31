get_ipython().magic('matplotlib inline')

import seaborn as sns
import numpy as np
from scipy.stats import beta, gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

beta_dist = beta(2, 2)


def gen_x_draws(k):
    """
    Returns a flat array containing k independent draws from the
    distribution of X, the underlying random variable.  This distribution is
    itself a convex combination of three beta distributions.
    """
    bdraws = beta_dist.rvs((3, k))
    # == Transform rows, so each represents a different distribution == #
    bdraws[0, :] -= 0.5
    bdraws[1, :] += 0.5
    bdraws[2, :] -= 1.0
    # == Set X[i] = bdraws[j, i], where j is a random draw from {0, 1, 2} == #
    js = np.random.random_integers(0, 2, size=k)
    X = bdraws[js, np.arange(k)]
    # == Rescale, so that the random variable is zero mean == #
    m, sigma = X.mean(), X.std()
    return (X - m) / sigma

nmax = 5
reps = 100000
ns = list(range(1, nmax + 1))
nsy = ['$Q_{}$'.format(i) for i in ns]

# == Form a matrix Z such that each column is reps independent draws of X == #
Z = np.empty((reps, nmax))
for i in range(nmax):
    Z[:, i] = gen_x_draws(reps)
# == Take cumulative sum across columns
S = Z.cumsum(axis=1)
# == Multiply j-th column by sqrt j == #
Y = (1 / np.sqrt(ns)) * S

# == Plot == #

fig = plt.figure(figsize=(14, 10))
ax = fig.gca(projection='3d')
ax.view_init(elev=20., azim=220)

a, b = -3, 3
gs = 100
xs = np.linspace(a, b, gs)

# == Build verts == #
greys = np.linspace(0.3, 0.7, nmax)
verts = []
for n in ns:
    density = gaussian_kde(Y[:, n-1])
    ys = density(xs)
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts)
#poly = PolyCollection(verts, facecolors=[str(g) for g in greys])
poly.set_alpha(0.85)
ax.add_collection3d(poly, zs=ns, zdir='x')

# ax.text(np.mean(rhos), a-1.4, -0.02, r'$\beta$', fontsize=16)
# ax.text(np.max(rhos)+0.016, (a+b)/2, -0.02, r'$\log(y)$', fontsize=16)
ax.set_xlim3d(1, nmax)
ax.set_xticks(ns)
ax.set_xticklabels(nsy)
ax.set_yticks([0])
ax.set_ylim3d(a, b)
ax.set_zlim3d(0, 0.4)
ax.set_zticks((0.2, 0.4))
plt.show()



