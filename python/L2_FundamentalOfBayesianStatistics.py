get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from scipy import stats
from scipy.misc import comb
from scipy.special import beta

t, T, s = sp.symbols('theta, T, s')

# Create the function symbolically
likelihood = (t**s)*(1-t)**(T-s)

# Convert it to a Numpy-callable function
_likelihood = sp.lambdify((t,T,s), likelihood, modules='numpy')

# Prior
a = 2
b = 2
Theta= np.linspace(0,1, 1000)


#Data generation
N=100
theta = 0.3
y = np.random.binomial(1, theta, N)



# Plot the prior
fig = plt.figure(figsize=(10,2))
ax = fig.add_subplot(111)
ax.plot(Theta, stats.beta(a, b).pdf(Theta), 'b');
ax.set(title='Prior Distribution')
ax.legend(['Prior']);



#Lilkelihood
likelihood = np.zeros(len(Theta))
for i, theta in enumerate(Theta):
    likelihood[i] = _likelihood(theta,N,y.sum())
    
# Plot the likelihood
fig = plt.figure(figsize=(10,2))
ax = fig.add_subplot(111)
ax.plot(Theta, likelihood, 'g');
ax.set(title='Likelihood')
ax.legend(['Prior']);    




# Posterior
a_hat = a + y.sum()
b_hat = b + N - y.sum()

# Plot the analytic posterior
fig = plt.figure(figsize=(10,2))
ax = fig.add_subplot(111)
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a_hat, b_hat).pdf(Theta), 'r');

# Plot the prior
ax.plot(X, stats.beta(a, b).pdf(X), 'g');

# Cleanup
ax.set(title='Posterior Distribution (Analytic)')
ax.legend(['Posterior (Analytic)', 'Prior']);



figsize(11, 9)
dist = stats.beta
alpha_0=10
beta_0=10
n_trials = [0, 10, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)

for k, N in enumerate(n_trials):
    sx = plt.subplot(len(n_trials) / 2, 2, k + 1)
    plt.xlabel("$p$, probability of heads")         if k in [0, len(n_trials) - 1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, alpha_0 + heads, beta_0 + N - heads)
    plt.plot(x, y, color="r",label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="r", alpha=0.1)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)
    plt.suptitle("Bayesian updating of posterior probabilities",
             y=1.02,
             fontsize=14)
    plt.tight_layout()

# Prior
a = 5
b = 5
Theta= np.linspace(0,1, 1000)


#Data generation
N=100
theta = 0.3
Y_hat=range(0,N)
y = np.random.binomial(1, theta, N)



#1.Prior
plt.figure(figsize=(10,2))
plt.plot(Theta, stats.beta(a, b).pdf(Theta), 'r');
plt.suptitle("Prior distribution")
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$p(\theta)$', fontsize=15)



#2.Prior predictive distribution
y_prior_predictive = np.zeros(N)
for i, y_hat in enumerate(Y_hat):
    y_prior_predictive[i] = comb(N,y_hat)*beta(a+y_hat, b+N-y_hat)/beta(a,b)
 
plt.figure(figsize=(10,2))
plt.plot(Y_hat, y_prior_predictive, 'or');
plt.suptitle("Prior predictive distribution")
plt.xlabel(r'$y$', fontsize=15)
plt.ylabel(r'$p(y)$', fontsize=15)


#3.Lilkelihood
likelihood = np.zeros(len(Theta))
for i, theta in enumerate(Theta):
    likelihood[i] = _likelihood(theta,N,y.sum())
    
plt.figure(figsize=(10,2))
plt.plot(Theta, likelihood, 'g');
plt.suptitle("Likelihood")
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$p(y|\theta)$', fontsize=20)



#4.Posterior distribution
a_hat = a + y.sum()
b_hat = b + N - y.sum()

# Plot the analytic posterior
plt.figure(figsize=(10,2))
plt.plot(Theta, stats.beta(a_hat, b_hat).pdf(Theta), 'r');
plt.suptitle("Posterior distribution")
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$p(\theta|y)$', fontsize=15)





#Posterior predictive distribution
y_prior_predictive = np.zeros(N)
for i, y_hat in enumerate(Y_hat):
    y_prior_predictive[i] = comb(N,y_hat)*beta(a_hat+y_hat, b_hat+N-y_hat)/beta(a_hat,b_hat)
plt.figure(figsize=(10,2))
plt.plot(Y_hat, y_prior_predictive, 'or');
plt.suptitle("Posterior predictive distribution")
plt.xlabel("$\hat{y}$", fontsize=20)
plt.ylabel("$p(\hat{y}|y)$", fontsize=20)



