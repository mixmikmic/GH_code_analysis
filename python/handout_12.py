import numpy as np
import scipy

class KarhunenLoeveExpansion(object):
    
    """
    A class representing the Karhunen Loeve Expansion of a Gaussian random field.
    It uses the Nystrom approximation to do it.
    
    Arguments:
        k      -     The covariance function.
        Xq     -     Quadrature points for the Nystrom approximation.
        wq     -     Quadrature weights for the Nystrom approximation.
        alpha  -     The percentage of the energy of the field that you want to keep.
        X      -     Observed inputs (optional).
        Y      -     Observed field values (optional).
    """
    
    def __init__(self, k, Xq=None, wq=None, nq=100, alpha=0.9, X=None, Y=None):
        self.k = k
        if Xq is None:
            if k.input_dim == 1:
                Xq = np.linspace(0, 1, nq)[:, None]
                wq = np.ones((nq, )) / nq
            elif k.input_dim == 2:
                nq = int(np.sqrt(nq))
                x = np.linspace(0, 1, nq)
                X1, X2 = np.meshgrid(x, x)
                Xq = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])
                wq = np.ones((nq ** 2, )) / nq ** 2
            else:
                raise NotImplementedError('For more than 2D, please supply quadrature points and weights.')
        self.Xq = Xq
        self.wq = wq
        self.k = k
        self.alpha = alpha
        self.X = X
        self.Y = Y
        # If we have some observed data, we need to use the posterior covariance
        if X is not None:
            gpr = GPy.models.GPRegression(X, Y[:, None], k)
            gpr.likelihood.variance = 1e-12
            self.gpr = gpr
            Kq = gpr.predict(Xq, full_cov=True)[1]
        else:
            Kq = k.K(Xq)
        B = np.einsum('ij,j->ij', Kq, wq)
        lam, v = scipy.linalg.eigh(B, overwrite_a=True)
        lam = lam[::-1]
        lam[lam <= 0.] = 0.
        energy = np.cumsum(lam) / np.sum(lam)
        i_end = np.arange(energy.shape[0])[energy > alpha][0] + 1
        lam = lam[:i_end]
        v = v[:, ::-1]
        v = v[:, :i_end]
        self.lam = lam
        self.sqrt_lam = np.sqrt(lam)
        self.v = v
        self.energy = energy
        self.num_xi = i_end
        
    def eval_phi(self, x):
        """
        Evaluate the eigenfunctions at x.
        """
        if self.X is not None:
            nq = self.Xq.shape[0]
            Xf = np.vstack([self.Xq, x])
            m, C = self.gpr.predict(Xf, full_cov=True)
            Kc = C[:nq, nq:].T
            self.tmp_mu = m[nq:, :].flatten()
        else:
            Kc = self.k.K(x, self.Xq)
            self.tmp_mu = 0.
        phi = np.einsum("i,ji,j,rj->ri", 1. / self.lam, self.v, self.wq**0.5, Kc)
        return phi
    
    def __call__(self, x, xi):
        """
        Evaluate the expansion at x and xi.
        """
        phi = self.eval_phi(x)
        return self.tmp_mu + np.dot(phi, xi * self.sqrt_lam)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import GPy
k = GPy.kern.RBF(1, lengthscale=0.1)
kle = KarhunenLoeveExpansion(k, nq=5, alpha=.9)
x = np.linspace(0, 1, 100)[:, None]
fig, ax = plt.subplots()
ax.plot(x, kle.eval_phi(x))
ax.set_xlabel('$x$')
ax.set_ylabel('$\phi_i(x)$')
fig, ax = plt.subplots()
ax.plot(kle.lam)
ax.set_xlabel('$i$')
ax.set_ylabel('$\lambda_i$');

x = np.linspace(0, 1, 100)[:, None]
fig, ax = plt.subplots()
for ell in [0.01, 0.05, 0.1, 0.2, 0.5]:
    k = GPy.kern.RBF(1, lengthscale=ell)
    kle = KarhunenLoeveExpansion(k, nq=100, alpha=.9)
    ax.plot(kle.lam[:5], '-x', markersize=5, markeredgewidth=2, label='$\ell={0:1.2f}$'.format(ell))
plt.legend(loc='best')
ax.set_xlabel('$i$')
ax.set_ylabel('$\lambda_i$');

k = GPy.kern.Exponential(1, lengthscale=0.1)
kle = KarhunenLoeveExpansion(k, nq=100, alpha=0.8)
x = np.linspace(0, 1, 100)[:, None]
fig, ax = plt.subplots()
for i in xrange(3):
    xi = np.random.randn(kle.num_xi)
    f = kle(x, xi)
    plt.plot(x, f, color=sns.color_palette()[0])

# Just generate some input/output pairs randomly...
np.random.seed(12345)
X = np.random.rand(3, 1)
Y = np.random.randn(3)
# X and Y are assumed to be observed

k = GPy.kern.RBF(1, lengthscale=0.1)
kle = KarhunenLoeveExpansion(k, nq=100, alpha=0.9, X=X, Y=Y)
x = np.linspace(0, 1, 100)[:, None]
fig, ax = plt.subplots()
ax.plot(x, kle.eval_phi(x))
ax.set_xlabel('$x$')
ax.set_ylabel('$\phi_i(x)$')
fig, ax = plt.subplots()
ax.plot(X, Y, 'kx', markeredgewidth=2)
for i in xrange(3):
    xi = np.random.randn(kle.num_xi)
    f = kle(x, xi)
    plt.plot(x, f, color=sns.color_palette()[0])

k = GPy.kern.RBF(2, lengthscale=0.1)
#X = np.random.rand(3, 2)
#Y = np.random.randn(3)
kle = KarhunenLoeveExpansion(k, nq=100, alpha=0.9)#, X=X, Y=Y)
x = np.linspace(0, 1, 32)
X1, X2 = np.meshgrid(x, x)
X_all = np.hstack([X1.flatten()[:, None], X2.flatten()[:, None]])
print 'Number of terms:', kle.num_xi
# Let's look at them
Phi = kle.eval_phi(X_all)
for i in xrange(5):
    fig, ax = plt.subplots()
    c = ax.contourf(X1, X2, Phi[:, i].reshape(X1.shape))
    #ax.plot(X[:, 0], X[:, 1], 'rx', markeredgewidth=2)
    plt.colorbar(c)

