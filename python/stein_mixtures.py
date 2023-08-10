from IPython.display import Image
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import autograd.numpy as np
from autograd import grad

get_ipython().run_line_magic('matplotlib', 'inline')

# AdaM: Adaptive Moments Optimizer
## Params
### alpha0: base learning rate
### grad: current gradient
### adam_values: dictionary containing moment estimates

def get_AdaM_update(alpha_0, grad, adam_values, b1=.95, b2=.999, e=1e-8):
    adam_values['t'] += 1

    # update mean                                                                                                                                                                                                     
    adam_values['mean'] = b1 * adam_values['mean'] + (1-b1) * grad
    m_hat = adam_values['mean'] / (1-b1**adam_values['t'])

    # update variance                                                                                                                                                                                                 
    adam_values['var'] = b2 * adam_values['var'] + (1-b2) * grad**2
    v_hat = adam_values['var'] / (1-b2**adam_values['t'])

    return alpha_0 * m_hat/(np.sqrt(v_hat) + e)

# Gaussian
def gaussPdf(x, params):
    # params: {'mu': mean, 'sigma': standard dev.}
    return (1./np.sqrt(2*np.pi*params['sigma']**2)) * np.exp((-.5/params['sigma']**2) * np.sum((x-params['mu'])**2))

# 2D Gaussian Mixture
def logGaussMixPDF(x, params):
    # params: {'pi': list of weights, 'mu': list of means, 'sigma': list of standard devs}
    return np.log(params['pi'][0] * gaussPdf(x, {'mu':params['mu'][0], 'sigma':params['sigma'][0]})             + params['pi'][1] * gaussPdf(x, {'mu':params['mu'][1], 'sigma':params['sigma'][1]}))

true_posterior_params = {
    'mu': [-4,3],
    'sigma': [1, 3],
    'pi': [.3, .7]
}

plt.figure()

theta_grid = np.linspace(-10, 10, 1000)

probs_true = [np.exp(logGaussMixPDF(z, true_posterior_params)) for z in theta_grid]
plt.plot(theta_grid, probs_true, 'b-', linewidth=5, label="True Posterior")

plt.xlabel(r"$\theta$")
plt.xlim([-10,10])
plt.ylim([0,.25])
plt.legend()

plt.show()

# Entropy of Gaussian
def gaussEntropy(log_sigma):
    return .5 * (np.log(2*np.pi*np.e) + 2.*log_sigma)

# Function for sampling from Gaussian location-scale form
def sample_from_Gauss(mu, log_sigma):
    e = np.random.normal()
    return mu + np.exp(log_sigma) * e, e


### GET DERIVATIVES ###

# d log p(X, \theta) / d \theta
logModel = logGaussMixPDF
dLogModel_dTheta = grad(logModel)

# d theta / d log_sigma
### we'll implement this ourselves

# d entropy / d log_sigma
dEntropy_dLogSigma = grad(gaussEntropy)

### INIT VARIATIONAL PARAMS 
phi = {'mu':-5., 'log_sigma':0.}


### ELBO OPTIMIZATION
maxEpochs = 500
learning_rate = .1
adam_values = {'mu':{'mean': 0., 'var': 0., 't': 0}, 'log_sigma':{'mean': 0., 'var': 0., 't': 0}}
n_samples = 10

for epochIdx in range(maxEpochs):
    
    elbo_grad_mu, elbo_grad_log_sigma = 0., 0.
    for s in range(n_samples):
        
        theta_hat, rand_seed = sample_from_Gauss(phi['mu'], phi['log_sigma'])
        dModel_dTheta = dLogModel_dTheta(theta_hat, true_posterior_params)
        
        elbo_grad_mu += 1./n_samples * dModel_dTheta * 1.
        elbo_grad_log_sigma += 1./n_samples * dModel_dTheta * rand_seed * np.exp(phi['log_sigma'])
        
    elbo_grad_log_sigma += dEntropy_dLogSigma(phi['log_sigma'])
        
    phi['mu'] += get_AdaM_update(learning_rate, elbo_grad_mu, adam_values['mu'])  
    phi['log_sigma'] += get_AdaM_update(learning_rate, elbo_grad_log_sigma, adam_values['log_sigma']) 
        
print phi

probs_approx = [gaussPdf(z, {'mu':phi['mu'], 'sigma':np.exp(phi['log_sigma'])}) for z in theta_grid] 
    
plt.figure()

plt.plot(theta_grid, probs_true, 'b-', linewidth=7, label="True Posterior")
plt.plot(theta_grid, probs_approx, '-r', linewidth=5, label="Variational Approx.")

plt.xlabel(r"$\theta$")
plt.xlim([-10,10])
plt.ylim([0,.25])
plt.legend()

plt.show()

# Kernel: Radial Basis Function 
def rbf(x1, x2, params={'lengthScale': 5.}):
    return np.exp((-.5/params['lengthScale']) * np.sum((x1-x2)**2))


# SVGD Operator 
def steinOp(theta_particles, dLogModel, kernel, dKernel):
    K = len(theta_particles)
    
    # precompute model derivative w.r.t. each particle
    dModel_dThetas = [0.] * K
    for k in range(K):
        dModel_dThetas[k] = dLogModel(theta_particles[k], true_posterior_params)
    
    # compute each particle's update
    particle_updates = [0.] * K
    for k in range(K):
        for j in range(K):
            particle_updates[k] += kernel(theta_particles[j], theta_particles[k]) * dModel_dThetas[j]                                     + dKernel(theta_particles[j], theta_particles[k])
        particle_updates[k] /= K
        
    return particle_updates

### INIT VARIATIONAL PARTICLES 
n_particles = 10
theta_particles = [np.random.normal(loc=-5.) for k in range(n_particles)]


### STEIN VARIATIONAL GRADIENT DESCENT
maxEpochs = 500
learning_rate = .1
adam_values = [{'mean': 0., 'var': 0., 't': 0} for k in range(n_particles)]

for epochIdx in range(maxEpochs):
    
    particle_updates = steinOp(theta_particles, dLogModel_dTheta, rbf, grad(rbf))
    
    for k in range(n_particles):
        theta_particles[k] += get_AdaM_update(learning_rate, particle_updates[k], adam_values[k])  
        
print theta_particles

theta_particles.sort()
probs_approx = [np.exp(logModel(z, true_posterior_params)) for z in theta_particles]   
    
plt.figure()

plt.plot(theta_grid, probs_true, 'b-', linewidth=7, label="True Posterior")
plt.plot(theta_particles, probs_approx, 'sr-', markersize=14, mew=0, linewidth=5, label="Variational Approx.")

plt.xlabel(r"$\theta$")
plt.xlim([-10,10])
plt.ylim([0,.25])
plt.legend()

plt.show()

# Kernel: Probability Product Kernel 
# http://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf
def prob_prod(z1_mu, z1_log_sigma, z2_mu, z2_log_sigma, rho=1.75):
    
    z1_sigma = np.exp(2*z1_log_sigma)
    z2_sigma = np.exp(2*z2_log_sigma)
    sigma_star = 1./z1_sigma + 1./z2_sigma
    mu_star = z1_mu/z1_sigma + z2_mu/z2_sigma

    return np.exp( -rho/2. * ((z1_mu**2)/z1_sigma + (z2_mu**2)/z2_sigma - (mu_star**2)/sigma_star ) )


### GET KERNEL DERIVATIVES
kernel_grad_fns = {}
kernel_grad_fns['mu'] = grad(prob_prod)
kernel_grad_fns['log_sigma'] = grad(lambda log_sigma1, mu1, mu2, log_sigma2: prob_prod(mu1, log_sigma1, mu2, log_sigma2))

### GET Q(\theta) DERIVATIVES
logQ_grad_fns = {}
logQ_grad_fns['x'] = grad(lambda x, mu, log_sigma: np.log(gaussPdf(x, {'mu':mu, 'sigma':np.exp(log_sigma)})))
logQ_grad_fns['mu'] = grad(lambda mu, x, log_sigma: np.log(gaussPdf(x, {'mu':mu, 'sigma':np.exp(log_sigma)})))
logQ_grad_fns['log_sigma'] = grad(lambda log_sigma, x, mu: np.log(gaussPdf(x, {'mu':mu, 'sigma':np.exp(log_sigma)})))

# Stein Mixture Operator 
def steinMixOp(phi_particles, dLogModel, dLogQ, kernel, dKernel):
    K = len(phi_particles)
    
    # precompute model derivative w.r.t. each particle
    # assumes just ONE sample is taken
    grad_mu, grad_logSig = [0.]*K, [0.]*K
    for k in range(K):
        
        theta_hat, rand_seed = sample_from_Gauss(phi_particles[k]['mu'], phi_particles[k]['log_sigma'])
        dModel_dTheta = dLogModel(theta_hat, true_posterior_params)
        dTheta_dLogSig = rand_seed * np.exp(phi_particles[k]['log_sigma'])
        
        grad_mu[k] += dModel_dTheta * 1. 
        grad_mu[k] += -dLogQ['x'](theta_hat, phi_particles[k]['mu'], phi_particles[k]['log_sigma']) * 1
        grad_mu[k] += -dLogQ['mu'](phi_particles[k]['mu'], theta_hat, phi_particles[k]['log_sigma'])
        
        grad_logSig[k] += dModel_dTheta * dTheta_dLogSig 
        grad_logSig[k] += -dLogQ['x'](theta_hat, phi_particles[k]['mu'], phi_particles[k]['log_sigma']) * dTheta_dLogSig
        grad_logSig[k] += -dLogQ['log_sigma'](phi_particles[k]['log_sigma'], theta_hat, phi_particles[k]['mu'])
        
        
    # compute each particle's update
    particle_updates_mu, particle_updates_logSig = [0.]*K, [0.]*K 
    for k in range(K):
        
        mu_k, logSig_k = phi_particles[k]['mu'], phi_particles[k]['log_sigma']
        for j in range(K):
            
            mu_j, logSig_j = phi_particles[j]['mu'], phi_particles[j]['log_sigma']
            
            particle_updates_mu[k] += kernel(mu_j, logSig_j, mu_k, logSig_k) * grad_mu[j]                                     + dKernel['mu'](mu_j, logSig_j, mu_k, logSig_k)
            particle_updates_logSig[k] += kernel(mu_j, logSig_j, mu_k, logSig_k) * grad_logSig[j]                                     + dKernel['log_sigma'](logSig_j, mu_j, mu_k, logSig_k)
                
        particle_updates_mu[k] /= K
        particle_updates_logSig[k] /= K
        
    return particle_updates_mu, particle_updates_logSig

### INIT VARIATIONAL PARTICLES 
n_particles = 3
phi_particles = [{'mu':np.random.normal(), 'log_sigma':0.} for k in range(n_particles)]


### STEIN MIXTURES
maxEpochs = 500
mu_learning_rate = .03
logSig_learning_rate = .003
adam_values = [{'mu':{'mean': 0., 'var': 0., 't': 0}, 'log_sigma':{'mean': 0., 'var': 0., 't': 0}} for k in range(n_particles)]

for epochIdx in range(maxEpochs):
    
    particle_updates = steinMixOp(phi_particles, dLogModel_dTheta, logQ_grad_fns, prob_prod, kernel_grad_fns)
    
    for k in range(n_particles):
        phi_particles[k]['mu'] += get_AdaM_update(mu_learning_rate, particle_updates[0][k], adam_values[k]['mu'])
        phi_particles[k]['log_sigma'] += get_AdaM_update(logSig_learning_rate, particle_updates[0][k], adam_values[k]['log_sigma'])
        
print phi_particles

plt.figure()

plt.plot(theta_grid, probs_true, 'b-', linewidth=7, label="True Posterior")

probs_approx = []
for k in range(n_particles):
    probs_approx.append([gaussPdf(z, {'mu':phi_particles[k]['mu'], 'sigma':np.exp(phi_particles[k]['log_sigma'])}) for z in theta_grid]) 
    plt.plot(theta_grid, probs_approx[-1], '--k', linewidth=2, label="Component #%d" %(k+1))
    
full_approx = 1./n_particles * np.array(probs_approx).sum(axis=0)
plt.plot(theta_grid, full_approx, '-r', linewidth=5, label="Full Approximation")

plt.xlabel(r"$\theta$")
plt.xlim([-10,10])
plt.ylim([0,.25])
plt.legend()

plt.show()

