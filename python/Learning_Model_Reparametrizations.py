import cPickle as cp

from IPython.display import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import bivariate_normal

import autograd.numpy as np
from autograd import grad
get_ipython().run_line_magic('matplotlib', 'inline')

dim = 2
plot_x_lim = [-6, 6]
plot_y_lim = [-30, 10]
x = np.arange(plot_x_lim[0], plot_x_lim[1], 0.5)
y = np.arange(plot_y_lim[0], plot_y_lim[1], 0.5)
X, Y = np.meshgrid(x, y)

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

### Make Dataset
N = 1000
data = np.random.normal(size=(N, dim))
z1 = 1.25*data[:,0]
z2 = data[:,1]/1.25 - 1.5*(z1**2 + 1.25)
data = np.hstack([z1[np.newaxis].T, z2[np.newaxis].T])
np.random.shuffle(data)

plt.scatter(data[:,0], data[:,1], marker='x', s=25, c='k', alpha=.7)

plt.xlim([plot_x_lim[0], plot_x_lim[1]])
plt.ylim([plot_y_lim[0], plot_y_lim[1]])
plt.show()

# Gaussian
def logGaussPdf(x, params):
    # params: {'mu': mean, 'sigma': standard dev.}
    return  -.5/params['sigma']**2 * (x-params['mu'])**2 + -.5 * np.log(2*np.pi*params['sigma']**2)

# Gaussian model
def logModel(data, theta, logSigma):
    return np.sum(logGaussPdf(data, {'mu':theta, 'sigma': np.exp(logSigma)}))

# Gaussian model with lower triangular inv transform
def logModel_w_transform(data, mu, logL, e, logSigma):
    L = np.exp(logL)
    theta = mu + np.dot(e, L)
    return logModel(data, theta, logSigma) + np.sum(np.log(np.diag(L)))

dLogModelWithTransform_dMu = grad(lambda mu, data, logL, e, logSigma: logModel_w_transform(data, mu, logL, e, logSigma))

dLogModelWithTransform_dLogL = grad(lambda logL, data, mu, e, logSigma: logModel_w_transform(data, mu, logL, e, logSigma))

dLogModel_dLogSigma = grad(lambda logSigma, data, theta: logModel(data, theta, logSigma))

def run_MH(data, logModel_w_transform, model_params, reparam_params, e0, n_MCMC_its):
    samples = []
    
    for idx in range(n_MCMC_its):
        proposal = np.random.normal(loc=e0, scale=1, size=(1,2))
        
        log_ratio = logModel_w_transform(data, reparam_params['mu'], reparam_params['logL'], proposal, model_params['logSigma'])                 - logModel_w_transform(data, reparam_params['mu'], reparam_params['logL'], e0, model_params['logSigma'])
        log_correction = np.sum(logGaussPdf(e0, {'mu':proposal, 'sigma':1.})) - np.sum(logGaussPdf(proposal, {'mu':e0, 'sigma':1.}))
        
        accept_prob = np.minimum(1., np.exp(log_ratio + log_correction))
        uni_prob = np.random.uniform(low=0., high=1.)
        
        if uni_prob <= accept_prob:
            samples.append(proposal)
            e0 = proposal
            
    return samples

def run_implicitMCMC_VI(data, model_params, reparam_params, lr=.001, n_epochs=100, nChains=20, n_MCMC_its=100):
    
    adam_values = {'mu':{'mean': 0., 'var': 0., 't': 0}, 
                    'logL':{'mean': 0., 'var': 0., 't': 0},
                    'logSigma':{'mean': 0., 'var': 0., 't': 0},
                   }
    
    for it_idx in range(n_epochs):
        
        # update the reparametrization parameters
        e_samples = []
        for s in range(nChains):
            e0 = np.array([[np.random.normal(scale=np.sqrt(2)), np.random.normal(loc=-25, scale=np.sqrt(3))]])
            e_samples.append(run_MH(data, logModel_w_transform, model_params, reparam_params, e0, n_MCMC_its)[-1])
        
        grad_mu = 0.
        grad_logL = 0.
        for e in e_samples:
            grad_mu += dLogModelWithTransform_dMu(reparam_params['mu'], data, reparam_params['logL'], e, model_params['logSigma']) / nChains
            grad_logL += dLogModelWithTransform_dLogL(reparam_params['logL'], data, reparam_params['mu'], e, model_params['logSigma']) / nChains
        
        reparam_params['mu'] += get_AdaM_update(lr, grad_mu/len(e_samples), adam_values['mu']) 
        reparam_params['logL'] += np.tril( get_AdaM_update(lr, grad_logL, adam_values['logL']) )
        
        # update model params
        thetas = []
        for s in range(nChains):
            e0 = np.array([[np.random.normal(scale=np.sqrt(2)), np.random.normal(loc=-25, scale=np.sqrt(3))]])
            e_samples = run_MH(data, logModel_w_transform, model_params, reparam_params, e0, n_MCMC_its)
            thetas.append( reparam_params['mu'] + np.dot(e_samples[-1], np.exp(reparam_params['logL'])) )
        
        grad_logSig = 0.
        for theta in thetas:
            grad_logSig += dLogModel_dLogSigma(model_params['logSigma'], data, theta) / nChains
        
        model_params['logSigma'] += get_AdaM_update(lr, grad_logSig, adam_values['logSigma'])
        
        # check data likelihood
        print "Epoch #%d, LL: %.4f" %(it_idx+1, np.mean([logModel(data, theta, model_params['logSigma']) for theta in thetas]))

    # get one more round of samples
    final_e_samples = []
    for s in range(500):
        e0 = np.array([[np.random.normal(scale=2), np.random.normal(loc=-20, scale=3)]])
        final_e_samples.append(run_MH(data, logModel_w_transform, model_params, reparam_params, e0, n_MCMC_its)[-1])
    
    return model_params, reparam_params, final_e_samples

# INIT PARAMS
model_params = {'logSigma': np.zeros((1,2))}
reparam_params = {'logL': np.eye(2)*.01, 'mu': np.zeros((1,2))}


maxEpochs = 100
learning_rate = .03
nChains = 50
nMCMCSteps = 50

model_params, reparam_params, final_e_samples = run_implicitMCMC_VI(data, model_params, reparam_params,                                                                     lr=learning_rate, n_epochs=maxEpochs,                                                                    nChains=nChains, n_MCMC_its=nMCMCSteps)

final_theta_samples = [reparam_params['mu'] + np.dot(e, np.exp(reparam_params['logL'])) for e in final_e_samples]

x_samples = []
for theta in final_theta_samples:
    for s in range(1):
        x_samples.append( theta + np.random.normal(size=(1,2))*np.exp(model_params['logSigma']) ) 

final_theta_samples = np.array(final_theta_samples)[:,0,:]
x_samples = np.array(x_samples)[:,0,:]

Z = bivariate_normal(X, Y, sigmax=np.sqrt(2), sigmay=np.sqrt(3), mux=0, muy=-25)
plt.contour(X,Y,Z, colors='blue', linewidths=3)

plt.scatter(data[:,0], data[:,1], marker='x', s=25, c='k', alpha=.7)
plt.scatter(x_samples[:,0], x_samples[:,1], marker='x', s=25, c='r', alpha=.7)

plt.xlim([plot_x_lim[0], plot_x_lim[1]])
plt.ylim([plot_y_lim[0], plot_y_lim[1]])
plt.show()

