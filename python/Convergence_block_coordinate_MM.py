import numpy as np
import scipy as sp
import scipy.special
import scipy.linalg
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

def Do_Kalman_Likelihood(y, sigma2obs, sigma2e, x0):
    """MAP solution, inverse covariance matrix, and marginal loglikelihood of state-space model

    :param y: Observations (K,)
    :param sigma2obs: Variance of observation noise (vector)
    :param sigma2e: Variance of process noise (vector: sigma2e[i] variance of x[i] - x[i-1])
    :param x0: Initial value of state
    :return: x_map, L, marginal_loglikelihood, joint_loglikelihood
    """
    # Build diagonals of information matrix
    assert(len(y) == len(sigma2obs) and len(y) == len(sigma2e))
    D = 1. / sigma2obs + 1. / sigma2e + np.concatenate((1. / sigma2e[1:], [0.]))
    B = -1. / sigma2e
    B[0] = 0.
    
    # Find Cholesky factorization of Hessian
    U = sp.linalg.cholesky_banded((B, D), lower=False)

    # Solve linear system
    G = y / sigma2obs
    G[0] += x0 / sigma2e[0]
    x_map = sp.linalg.cho_solve_banded([U, False], G)

    # Compute joint and marginal probabilities
    joint_loglikelihood = -.5 * ( np.sum(np.log(2*np.pi*sigma2e)) + np.sum(np.diff(x_map)**2 / sigma2e[1:])
                                                                  + (x_map[0] - x0)**2 / sigma2e[0]
                                + np.sum(np.log(2*np.pi*sigma2obs)) + np.sum((y - x_map)**2 / sigma2obs) )
    marginal_loglikelihood = len(y)/2. * np.log(2*np.pi) + joint_loglikelihood - np.sum(np.log(U[-1]))
    return x_map, U, marginal_loglikelihood, joint_loglikelihood

def Do_MM(DN, c, lam, sigma2x, sigma2x0, x0, sigma2z0, z0, max_iter=100, tol=1e-6):
    """Find MAP solution of 2d model using MM

    :param dN: Observations (R,K,)
    :param c: Radius of Huber approximation to L1 norm
    :param lam: Coefficient of L1 penalization of changes along z
    :param sigma2x: Variance of x process noise
    :param sigma2x0: Variance of initial state x
    :param x0: Initial value of state x
    :param sigma2z0: Variance of initial state z
    :param z0: Initial value of state z
    :return: x_map, z_map, loglikelihood_map
    """
    # Initialize
    z = np.zeros((DN.shape[0],))
    x = np.zeros((DN.shape[1],))
    xz = x[None,:] + z[:,None]
    joint_loglikel = np.zeros(max_iter)
    # Iterate
    for i in range(max_iter):
        # Block MM over z
        B = 1. / np.sum(1. / np.maximum(4., 2. + np.abs(xz)), axis=1)
        A = np.sum(1. / (1. + np.exp(-xz)) - DN, axis=1)
        sigma2z = np.concatenate(([sigma2z0], np.maximum(c, np.abs(np.diff(z))) / lam))
        z = Do_Kalman_Likelihood(z - A*B, B, sigma2z, z0)[0]
        xz = x[None,:] + z[:,None]
        # Block MM over x
        B = 1. / np.sum(1. / np.maximum(4., 2. + np.abs(xz)), axis=0)
        A = np.sum(1. / (1. + np.exp(-xz)) - DN, axis=0)
        sigma2x_ = np.concatenate(([sigma2x0], np.ones(len(x)-1) * sigma2x))
        x = Do_Kalman_Likelihood(x - A*B, B, sigma2x_, x0)[0]
        xz = x[None,:] + z[:,None]
        # Compute likelihood
        joint_loglikel[i] = ( np.sum(DN * xz - np.log(1. + np.exp(xz))) - .5 * (
                           (len(x) - 1) * np.log(2*np.pi*sigma2x) + np.sum(np.diff(x)**2 / sigma2x)
                           + np.log(2*np.pi*sigma2x0) + (x[0] - x0)**2 / sigma2x0 
                           + np.log(2*np.pi*sigma2z0) + (z[0] - z0)**2 / sigma2z0 )
                           + np.log(lam/2.) - lam * np.sum(np.abs(np.diff(z))) )
        # Check convergence
        if i > 0 and joint_loglikel[i] - joint_loglikel[i-1] < tol * np.abs(joint_loglikel[i]):
            return x, z, joint_loglikel[:i+1]
    return x, z, joint_loglikel

K = 200
R = 45
x = .1 + .2 * (np.arange(K) > 100)
z = .1 + .2 * (np.arange(R) > 15)
DN = (x[None,:] + z[:,None] > np.random.rand(R, K)).astype(float)
plt.figure()
plt.imshow(DN)

get_ipython().magic('time')
c = 1e-3
sigma2x = .1
sigma2x0 = 10
x0 = 0
sigma2z0 = 10
z0 = 0
lam = 10.
x, z, joint_loglikel = Do_MM(DN, c, lam, sigma2x, sigma2x0, x0, sigma2z0, z0)
plt.figure()
plt.plot(joint_loglikel)
print(joint_loglikel[-1])

plt.figure()
plt.plot(x)
plt.plot(z)



