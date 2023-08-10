from cops_and_robots.robo_tools.fusion.softmax import speed_model
get_ipython().magic('matplotlib inline')

sm = speed_model()
sm.plot(plot_classes=False)

from __future__ import division
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

mu = 0.3
sigma = 0.1
min_x = -5
max_x = 5
res = 10000

prior = norm(loc=mu, scale=sigma)
x_space = np.linspace(min_x, max_x, res)

# Plot the frozen distribution
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8)
ax.plot(x_space, prior.pdf(x_space), lw=2, label='frozen pdf', c='g')
ax.fill_between(x_space, 0, prior.pdf(x_space), alpha=0.2, facecolor='g')
ax.set_xlim([0,.4])
ax.set_ylim([0,10])
ax.set_title('Gaussian prior')

from numpy.linalg import inv
import pandas as pd
np.set_printoptions(precision=2, suppress=True)
pd.set_option('precision', 3)

# SETTINGS:
n_lc = 15  # number of convergence loops
measurement = 'Medium'
tolerance = 10 ** -3  # for convergence
max_EM_steps = 1000


# INPUT: Define input priors and initial values
prior_mu = np.zeros(1)
prior_sigma = np.array([[1]])
initial_alpha = 0.5
initial_xi = np.ones(4)

# Softmax values
m = 4
w = sm.weights
b = sm.biases
j = sm.class_labels.index(measurement)

# Preparation
xis = initial_xi
alpha = initial_alpha
mu_hat = prior_mu
sigma_hat = prior_sigma


# dataframe for debugging
df = pd.DataFrame({'Alpha': alpha,
                   'g_j' : np.nan,
                   'h_j' : np.nan,
                   'K_j' : np.nan,                   
                   'Mu': mu_hat[0],
                   'Sigma': sigma_hat[0][0],
                   'Xi': [xis],
                   })
def lambda_(xi_c):
    return 1 / (2 * xi_c) * ( (1 / (1 + np.exp(-xi_c))) - 0.5)


converged = False
EM_step = 0

while not converged and EM_step < max_EM_steps:
    ################################################################
    # STEP 1 - EXPECTATION
    ################################################################
    # PART A #######################################################

    # find g_j
    sum1 = 0
    for c in range(m):
        if c != j:
            sum1 += b[c]
    sum2 = 0
    for c in range(m):
        sum2 = xis[c] / 2             + lambda_(xis[c]) * (xis[c] ** 2 - (b[c] - alpha) ** 2)             - np.log(1 + np.exp(xis[c]))
    g_j = 0.5 *(b[j] - sum1) + alpha * (m / 2 - 1) + sum2

    # find h_j
    sum1 = 0
    for c in range(m):
        if c != j:
            sum1 += w[c]
    sum2 = 0
    for c in range(m):
        sum2 += lambda_(xis[c]) * (alpha - b[c]) * w[c]
    h_j = 0.5 * (w[j] - sum1) + 2 * sum2

    # find K_j
    sum1 = 0
    for c in range(m):
        sum1 += lambda_(xis[c]) * w[c].T .dot (w[c])

    K_j = 2 * sum1
        
    K_p = inv(prior_sigma)
    g_p = -0.5 * (np.log( np.linalg.det(2 * np.pi * prior_sigma)))         + prior_mu.T .dot (K_p) .dot (prior_sigma)
    h_p = K_p .dot (prior_mu)

    g_l = g_p + g_j
    h_l = h_p + h_j
    K_l = K_p + K_j

    mu_hat = inv(K_l) .dot (h_l)
    sigma_hat = inv(K_l)
        
    # PART B #######################################################
    y_cs = np.zeros(m)
    y_cs_squared = np.zeros(m)
    for c in range(m):
        y_cs[c] = w[c].T .dot (mu_hat) + b[c]
        y_cs_squared[c] = w[c].T .dot (sigma_hat + mu_hat .dot (mu_hat.T)) .dot (w[c])             + 2 * w[c].T .dot (mu_hat) * b[c] + b[c] ** 2

    ################################################################
    # STEP 2 - MAXIMIZATION
    ################################################################
    for i in range(n_lc):

        # PART A #######################################################
        # Find xi_cs
        for c in range(m):
            xis[c] = np.sqrt(y_cs_squared[c] + alpha ** 2 - 2 * alpha * y_cs[c])

        # PART B #######################################################
        # Find alpha
        num_sum = 0
        den_sum = 0
        for c in range(m):
            num_sum += lambda_(xis[c]) * y_cs[c]
            den_sum += lambda_(xis[c])
        alpha = ((m - 2) / 4 + num_sum) / den_sum

    ################################################################
    # STEP 3 - CONVERGENCE CHECK
    ################################################################    
    
    new_df = pd.DataFrame([[alpha, g_j, h_j, K_j, mu_hat, sigma_hat, 
                            [xis]]],
                          columns=('Alpha','g_j','h_j','K_j','Mu','Sigma',
                                   'Xi',))
    df = df.append(new_df, ignore_index=True)
    EM_step += 1
# df

#plot results
mu_post = mu_hat[0]
sigma_post = np.sqrt(sigma_hat[0][0])

print('Mu and sigma found to be {} and {}, respectively.'.format(mu_hat[0],sigma_hat[0][0]))

ax = sm.plot_class(measurement_i, fill_between=False)
posterior = norm(loc=mu_post, scale=sigma_post)
ax.plot(x_space, posterior.pdf(x_space), lw=2, label='posterior pdf', c='b')
ax.fill_between(x_space, 0, posterior.pdf(x_space), alpha=0.2, facecolor='b')
ax.plot(x_space, prior.pdf(x_space), lw=1, label='prior pdf', c='g')

ax.set_title('Posterior distribtuion')
ax.legend()
ax.set_xlim([0, 0.4])
ax.set_ylim([0, 7])
plt.show()



measurement = 'Slow'
measurement_i = sm.class_labels.index(measurement)

dx = (max_x - min_x)/res

normalizer = 0
for x in x_space:
    lh = sm.probs_at_state(x, measurement)
    if np.isnan(lh):
        lh = 1.00
    normalizer += lh  * gaussian.pdf(x)
normalizer *= dx
    
posterior = np.zeros_like(x_space)
for i, x in enumerate(x_space):
    lh = sm.probs_at_state(x, measurement)
    if np.isnan(lh):
        lh = 1.00
    posterior[i] = lh * gaussian.pdf(x) / normalizer
    
ax = sm.plot_class(measurement_i, fill_between=False)
ax.plot(x_space, posterior, lw=3, label='posterior pdf', c='b')
ax.fill_between(x_space, 0, posterior, alpha=0.2, facecolor='b')
ax.plot(x_space, prior.pdf(x_space), lw=1, label='prior pdf', c='g')

ax.set_title('Posterior distribtuion')
ax.legend()
ax.set_xlim([0, 0.4])
plt.show()



from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

