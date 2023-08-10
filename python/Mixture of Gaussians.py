import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
get_ipython().magic('matplotlib notebook')

mu_vals, sigma_vals = [[2, 2], [2, -2], [-2, -2], [-2, 2]], [[[.1, 0], [0, .1]] for i in range(4)]

samples = []
for mu, sigma in zip(mu_vals, sigma_vals):
    samples.append(np.random.multivariate_normal(mu, sigma, 500))

def initialParams(K, muSpread, sigmaWidth):
    means = np.array([np.random.uniform(low=-muSpread, high=muSpread, size=2) for _ in range(K)])
    
    sigmas = np.array([sigmaWidth * np.eye(2) for _ in range(K)])
    
    pi = np.array([1/K] * K)
    
    return {'mu': means, 'sigma': sigmas, 'pi': pi, 'K': K}

def expectation(data, parameters):
    # Evaluate responsibilities
    N = len(data)
    K = parameters['K']
    mu = parameters['mu']
    sigma = parameters['sigma']
    pi = parameters['pi']
    
    gammas = np.empty((N, K))
    
    for n in range(N):
        x_n = data[n, :]
        samples = np.array([multivariateNormal(x_n, mu[j], sigma[j]) for j in range(K)])
        for k in range(K):
            gamma_num = pi[k] * samples[k]
            gamma_denom = np.sum(pi * samples)
            gamma_nk = gamma_num / gamma_denom

            gammas[n, k] = gamma_nk
        
    return gammas

def multivariateNormal(x, mu, sigma):
    # Sample a single point from N(x | mu, sigma)
    sigma_inv = np.linalg.inv(sigma)
    diff = x - mu
    sigma_sqrt = np.sqrt(np.linalg.det(sigma))
    n = len(mu)
    
    # Calculate inverse of coefficient
    Z = (2 * np.pi) ** (n / 2) * sigma_sqrt
    
    # Calculate sample
    sample = (1 / Z) * np.exp( -.5 * diff.T @ sigma_inv @ diff)
    
    return sample
    

def maximization(data, gammas, parameters):
    mu_old = parameters['mu']
    sigma_old = parameters['sigma']
    pi_old = parameters['pi']
    K = parameters['K']
    N = len(data)
    
    mu_new = np.zeros_like(mu_old)
    sigma_new = np.zeros_like(sigma_old)
    pi_new = np.empty_like(pi_old)

    for k in range(K):
        N_k = np.sum(gammas[:, k])
        # Calculate new pi values
        pi_new[k] = N_k / N
        for n in range(N):
            # Calculate new mu values
            mu_new[k] += (1 / N_k) * data[n, :] * gammas[n, k]
        for n in range(N):
            # Calculate new sigma values
            sigma_new[k] += (1 / N_k) * gammas[n, k] * np.outer((data[n, :] - mu_new[k]), (data[n, :] - mu_new[k]))
        
    parameters['mu'] = mu_new
    parameters['sigma'] = sigma_new
    parameters['pi'] = pi_new
    
    return parameters

def hasConverged(logLikelihood, prevLikelihood, threshold=.1):          
    if np.abs(logLikelihood - prevLikelihood) < threshold:
        return True
    return False

def calcLogLikelihood(data, parameters):
    N = len(data)
    K = parameters['K']
    mu = parameters['mu']
    sigma = parameters['sigma']
    pi = parameters['pi']
    logLikelihood = 0
    partialSum = 1

    for n in range(N):
        logLikelihood += np.log(partialSum)
        partialSum = 0
        for k in range(K):
            partialSum += pi[k] * multivariateNormal(data[n, :], mu[k], sigma[k])

        return logLikelihood
    
def plotEM(data, parameters, ax):
    colors = ['red', 'cyan', 'green', 'black']
    K = parameters['K']
    
    for k in range(K):
        mu = parameters['mu'][k]
        sigma = parameters['sigma'][k]
        color = k % len(colors)
        plotContour(mu, sigma, ax, color=colors[color], numContours=5)
        

def runEM(data, num_iters=10, parameters=None, plot=True):
    if not parameters:
        # Adjust initial parameter generation as desired.
        parameters = initialParams(4, 1.5, .4)
        prevLikelihood = 0
        logLikelihood = 1
        
    for _ in range(num_iters):
        gammas = expectation(data, parameters)
        parameters = maximization(data, gammas, parameters)
        prevLikelihood = logLikelihood
        logLikelihood = calcLogLikelihood(data, parameters)
        
    if plot:       
        plotEMResults(data, parameters['K'], parameters)
    
def plotContour(mu,sigma,ax,color='blue',numContours=3):
    eigvalues, eigvectors = np.linalg.eig(sigma)
    primaryEigvector = eigvectors[:,0]
    angle = computeRotation(primaryEigvector)
    isoProbContours = [Ellipse(mu,
                               l*np.sqrt(eigvalues[0]),
                               l*np.sqrt(eigvalues[1]),
                               alpha=0.3/l,color=color,
                              angle=angle) 
                       for l in range(1,numContours+1)]
    [ax.add_patch(isoProbContour) for isoProbContour in isoProbContours]

def computeRotation(vector):
    return (180/np.pi)*np.arctan2(vector[1],vector[0])

def dataScatter(data,color='grey'):
    plt.scatter(data[:,0],data[:,1],color=color,edgecolor=None,alpha=0.1)
    return

def plotEMResults(dataset,K,parameters):
    plt.figure()
    dataScatter(dataset)
    for idx in range(K):
        mu = parameters['mu'][idx]
        sigma = parameters['sigma'][idx]
        color = idx % (len(colors))
        plotContour(mu,sigma,plt.gca(),color=colors[color],numContours=5)

fig, ax = plt.subplots(1)
colors = ['red', 'cyan', 'green', 'black']
for sample, color in zip(samples, colors):
    ax.plot(sample[:, 0], sample[:, 1], '.',  c=color, alpha=0.3, markersize=10);

data = np.vstack((samples[0], samples[1]))
data = np.vstack((data, samples[2]))
data = np.vstack((data, samples[3]))

runEM(data)

