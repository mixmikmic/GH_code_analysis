import numpy as np
import scipy as sp
import scipy.special
import scipy.linalg
#import hips.distributions.polya_gamma as pg
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def Do_Kalman_Likelihood(y, sigma2obs, sigma2e):
    """MAP solution, inverse covariance matrix, and marginal loglikelihood of state-space model

    :param y: Observations (K,)
    :param sigma2obs: Variance of observation noise (can be scalar or vector)
    :param sigma2e: Variance of process noise
    :return: x_map, L, marginal_loglikelihood, joint_loglikelihood
    """
    # Build diagonals of information matrix
    sigma2obs *= np.ones(len(y))
    D = 1. / sigma2obs + 2. / sigma2e
    D[-1] = 1. / sigma2obs[-1] + 1. / sigma2e
    B = -np.ones(len(D)) / sigma2e
    B[0] = 0.
    
    # Solve, assuming x_init=0 for simplicity
    #L = sp.linalg.cholesky_banded((D, B), lower=True)
    U = sp.linalg.cholesky_banded((B, D), lower=False)

    x_map = sp.linalg.cho_solve_banded([U, False], y / sigma2obs)

    # Compute joint and marginal probabilities
    joint_loglikelihood = -.5 * ((np.sum(np.diff(x_map)**2) + x_map[0]**2) / sigma2e +
                                 np.sum((y - x_map)**2 / sigma2obs) +
                                 (len(y) * np.log(2*np.pi*sigma2e * 2*np.pi) + np.sum(np.log(sigma2obs))))
    marginal_loglikelihood = len(y)/2. * np.log(2*np.pi) + joint_loglikelihood - np.sum(np.log(U[-1]))
    return x_map, U, marginal_loglikelihood, joint_loglikelihood

# Simple test of Do_Kalman_Likelihood only (K=1, Gaussian)
import scipy.stats as stats
sigma2e = 0.1
y = 3.
sigma2obs = 10.
x_map, U, marginal_loglikelihood, joint_loglikelihood = Do_Kalman_Likelihood(y * np.ones(1), sigma2obs, sigma2e)
j = stats.norm.pdf(x_map, 0, np.sqrt(sigma2e)) * stats.norm.pdf(x_map, y, np.sqrt(sigma2obs))
m = np.sqrt(2*np.pi) * j / U[-1,0]
assert(abs(np.log(j) - joint_loglikelihood) < 1e-9)
assert(abs(np.log(m) - marginal_loglikelihood) < 1e-9)

def Do_Kalman_Likelihood_Bernoulli_LaplaceMAP(dN, sigma2e, tol=1e-8, trials=1.):
    """MAP solution, inverse covariance matrix, and marginal loglikelihood of state-space model
    computed using Laplace approximation around MAP state.

    :param dN: Observations (K,)
    :param sigma2e: Variance of process noise
    :param tol: Convergence criterion on the gradient of the log-likelihood
    :param trials: Number of trials for binomial observations (1 for Bernoulli)
    :return: x_map, U, marginal_loglikelihood, joint_loglikelihood
    """
    x = np.zeros(dN.shape)
    dN = dN.astype(float)
    while True:
        # Build gradient of joint
        d2x = np.convolve(x, [-1, 2, -1])[1:-1]
        d2x[-1] -= x[-1]
        G = -dN + trials * (1. / (1. + np.exp(-x))) + d2x / sigma2e
        # Build Hessian of joint
        D = trials / (np.exp(x) + 2. + np.exp(-x)) + 2. / sigma2e
        D[-1] -= 1. / sigma2e
        B = -np.ones(len(D)) / sigma2e
        B[0] = 0.
        U = sp.linalg.cholesky_banded((B, D), lower=False)
        # Check convergence
        if np.dot(G, G) < tol:
            x_map = x
            break
        # Update estimate of map
        x -= sp.linalg.cho_solve_banded([U, False], G)

    # Compute joint and marginal probabilities
    joint_loglikelihood = (np.sum(np.log(sp.special.binom(trials, dN)) + dN * x_map - trials * np.log(1 + np.exp(x_map))) -
                           .5 * ((np.sum(np.diff(x_map)**2) + x_map[0]**2) / sigma2e + len(dN) * np.log(2*np.pi*sigma2e)))
    marginal_loglikelihood = len(dN)/2. * np.log(2*np.pi) + joint_loglikelihood - np.sum(np.log(U[-1]))
    return x_map, U, marginal_loglikelihood, joint_loglikelihood

# Test Marginal_Likelihood using the Laplace approximation around the MAP

# Load thaldata
import pandas as pd
dN = pd.read_csv('thaldata.csv', header=None).values[0]#[500:540]#800]
trials = 50
sigma2e = 0.12


get_ipython().magic('time x_map_l, U_l, marginal_loglikelihood_l, joint_loglikelihood_l = Do_Kalman_Likelihood_Bernoulli_LaplaceMAP(dN, sigma2e, trials=trials)')
print(marginal_loglikelihood_l)
plt.plot(x_map_l)
plt.plot(dN)
plt.show()
plt.plot(np.exp(x_map_l)/(1+np.exp(x_map_l)))

s2e = np.arange(.01, 1., .01)
marginal_loglikelihood_l = np.zeros_like(s2e)
joint_loglikelihood_l = np.zeros_like(s2e)
for i in range(len(s2e)):
    marginal_loglikelihood_l[i], joint_loglikelihood_l[i] = Do_Kalman_Likelihood_Bernoulli_LaplaceMAP(dN, s2e[i], trials=trials)[2:]
plt.plot(s2e, marginal_loglikelihood_l)
#plt.plot(s2e, joint_loglikelihood_l)

def Conditional_on_one_axis(dN, sigma2e, x_map, axis, values, trials=1.):
    """Compute the joint probability as a function of one of the coordinates of the state 'x'
    keeping all other coordinates at the MAP solution. Returns both the exact value and the Laplace approximation.

    :param dN: Observations (K,)
    :param sigma2e: Variance of process noise
    :param x_map: MAP solution (e.g. computed by Do_Kalman_Likelihood_Bernoulli_LaplaceMAP)
    :param axis: Along which axis the section of the joint should be computed (e.g. 3 for x_3)
    :param values: Values of x[axis] for which the joint should be computed
    :param trials: Number of trials for binomial observations (1 for Bernoulli)
    :return: joint_loglikelihood, joint_loglikelihood_map
    """
    binom = np.sum(np.log(sp.special.binom(trials, dN)))
    # Build Hessian of joint
    D = trials / (np.exp(x_map) + 2 + np.exp(-x_map)) + 2. / sigma2e
    D[-1] -= 1. / sigma2e
    B = -np.ones(len(D)) / sigma2e
    B[-1] = 0.
    L = sp.linalg.cholesky_banded((D, B), lower=True)
    joint_loglikelihood_map = (binom + np.sum(dN * x_map - trials * np.log(1 + np.exp(x_map)))
                           -.5 * ((np.sum(np.diff(x_map)**2) + x_map[0]**2) / sigma2e + len(dN) * np.log(2*np.pi*sigma2e)))
    joint_loglikelihood_map = joint_loglikelihood_map - .5 * (values - x_map[axis])**2 * (L[0][axis]**2 + L[1][axis]**2)
    x = x_map.copy()
    joint_loglikelihood = np.zeros(values.shape)
    for i in range(len(values)):
        x[axis] = values[i]
        # Compute joint and marginal probabilities
        joint_loglikelihood[i] = (binom + np.sum(dN * x - trials * np.log(1 + np.exp(x))) -
                                  .5 * ((np.sum(np.diff(x)**2) + x[0]**2) / sigma2e + len(dN) * np.log(2*np.pi*sigma2e)))
    return joint_loglikelihood, joint_loglikelihood_map

values = np.arange(-6., 2., .01)
axis = 12
joint_loglikelihood, joint_loglikelihood_map = Conditional_on_one_axis(dN, sigma2e, x_map_l, axis, values, trials=trials)
plt.plot(values, joint_loglikelihood)
plt.plot(values, joint_loglikelihood_map)

def cov_from_chol_precision(U):
    """Given the Cholesky factorization (U) of the posterior precision matrix (J), with U^t * U = J,
    return the tridiagonal part of the covariance matrix.

    :param U: Cholesky factorization (U) of J, given as [0, A; D] where A is the upper diagonal and D the main diagonal
    :return: Cov_tri: Tridiagonal part of the covariance matrix returned as [0, C_i,i+1; C_ii; C_i+1,i, 0]
    """
    assert(U.shape[0] == 2 and U[0,0] == 0)
    A, D = U # Unpack matrix into first (above) diagonal and diagonal
    Cov_tri = np.zeros_like(U)
    C, V = Cov_tri # Obtain _views_ into the first diagonal and diagonal
    # Compute last element of diagonal
    V[-1] = 1. / (D[-1] ** 2)
    # Recursively compute other elements of main diagonal and first diagonal
    for i in range(len(D)-1, 0, -1):
        iD = 1. / D[i-1]
        iDA = iD * A[i]
        N = -iDA * V[i]
        C[i] = N
        V[i-1] = iD ** 2 - N * iDA
    return Cov_tri

def EM_fit_sigma2e_Gaussian(y, sigma2v, sigma2e_init, tol=1e-6, trials=1.):
    """Optimize sigma2e using the EM algorithm for a 1D linear-Gaussian state-space model.

    :param y: Observations (K,)
    :param sigma2e_init: Initial estimate of sigma2e
    :param tol: Convergence criterion for the EM
    :param trials: Number of trials for binomial observations (1 for Bernoulli)
    :return: x_map, U, marginal_loglikelihood, joint_loglikelihood
    """
    sigma2e_old = sigma2e_init
    while True:
        x_map, U, marginal_loglikelihood, _ = Do_Kalman_Likelihood(y, sigma2obs, sigma2e_old)
        #print(sigma2e, marginal_loglikelihood)
        Cov_tri = cov_from_chol_precision(U)
        sigma2e = (np.sum(Cov_tri[1]) + np.dot(x_map, x_map) # E[x_k^2]
                   + np.sum(Cov_tri[1,:-1]) + np.dot(x_map[:-1], x_map[:-1]) # E[x_{k-1}^2]
                   - 2 * np.sum(Cov_tri[0]) - 2 * np.dot(x_map[1:], x_map[:-1])) / len(dN) # E[x_{k-1} * x_k]
        if (abs(sigma2e - sigma2e_old) < tol): break
        sigma2e_old = sigma2e
        return sigma2e

trials = 1
get_ipython().magic('pylab')

# Generate complex-valued Gaussian random vector
k = linspace(0,200,6000)
K = k.shape[0]

# Real part is an oscillation with period K0
K0 = 12.5
c1m = cos(2*pi*(k-((K0)/4))/(K0))

# Imaginary part is linear
#c2m = k/100
c2m = cos(2*pi*(k-(8*K0/4))/(8*K0))

# normalize cluster 1 mean and culster 2 mean they have the same energy
#c2m = sqrt(var(c1m)/var(c2m))*c2m

#figure(1)
#subplot(2,1,1)
#plot(k,c1m)
#title('Clean observations')
#ylabel('$s_{Re}$',fontsize=20)
#subplot(2,1,2)
#plot(k,c2m)
#ylabel('$s_{Im}$',fontsize=20)
#xlabel('Time (s)',fontsize=16);

# Add noise
# Add Gaussian noise (based on real part, which is 0 mean)
snr = 10 # in dB

sigma2e1 = var(c1m)
sigma2e2 = var(c2m)

sigma2v1 = sigma2e1*10**(-snr/10)
sigma2v2 = sigma2e2*10**(-snr/10)


y1 = c1m + sqrt(sigma2v1)*randn(K) 
y2 = c2m + sqrt(sigma2v2)*randn(K)

figure(2)
subplot(2,1,1)
plot(k,y1)
ylabel('$y_1$',fontsize=20)
subplot(2,1,2)
plot(k,y2)
ylabel('$y_2$',fontsize=20)
xlabel('Time (s)',fontsize=16);

#print('Observation variance: %f' % sigma2v)


sigma2eVec = np.linspace(0.00001,0.05,100)
llhdKal = np.zeros((2,sigma2eVec.shape[0]))

for i in range(len(sigma2eVec)):
    x_map, U, llhdKal[0,i], _ = Do_Kalman_Likelihood(y1, sigma2v1, sigma2eVec[i])
    x_map, U, llhdKal[1,i], _ = Do_Kalman_Likelihood(y2, sigma2v2, sigma2eVec[i])


#print U[0,0]

#%time sigma2e_opt = EM_fit_sigma2e_Gaussian(y2, sigma2v,0.5, trials=trials)
#print(sigma2e_opt)

figure()
plot((sigma2eVec),llhdKal.T)

print var(diff(c1m))
print var(diff(c2m))
#plot((c1m))

print sigma2eVec[np.argmax(llhdKal[0])]
print sigma2eVec[np.argmax(llhdKal[1])]


x_map1, U, llhdKal1, _ = Do_Kalman_Likelihood(y1, sigma2v, sigma2eVec[np.argmax(llhdKal[0])])
x_map2, U, llhdKal2, _ = Do_Kalman_Likelihood(y2, sigma2v, sigma2eVec[np.argmax(llhdKal[1])])

figure
plot(x_map1)
plot(x_map2)



