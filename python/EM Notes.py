# data 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

class Distribution(object):
    def __init__(self, color):
        self.color = color

num_gaussians = 4
pi = np.array([0.1, 0.3, 0.2, 0.3])
num_samples = 10000

mu = ([1.7, .5],
       [2, 4],
       [0, 6],
       [5, 6]
     )
sigma = ([[.9, 0], [0, .5]],
         [[.4, .3], [.3, .5]],
         [[2, .7], [.2, .8]],
         [[.6, .6], [.3, .6]]
        )

distributions = {}
colors = ['r','g','b','y']
for i in range(num_gaussians):
    name = 'Sampled Distribution {}'.format(i + 1)
    distributions[name] = Distribution(colors[i])
    
    distributions[name].samples = np.random.multivariate_normal(
        mu[i], sigma[i], int(pi[i] * num_samples))
    
# Plot everything
fig, ax = plt.subplots()
for name, distribution in distributions.iteritems():
    ax.scatter(distribution.samples[:,0],
            distribution.samples[:,1],
            c=distribution.color,
            s=20,
            lw=0
            )
ax.set_title('Sampled distributions')

# Initial setup
K = 4  # <>But how do we know?
mu_hats = []
sigma_hats = []
pi_hats = []
for k in range(K):
    mu_hats.append(np.rand.randint(-10,10))
    sigma_hats.append(np.eye(2))
    pi_hat
    

from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

