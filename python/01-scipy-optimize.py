get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

# Silence warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'

from utils import beer_score

def inverse_beer_score(points):
    return -1. * beer_score(points)

budget = 50

bounds = [(2., 14.), (5., 80.)]

from scipy import optimize

optimize.fmin_l_bfgs_b(inverse_beer_score, (8., 40.),
                       bounds=bounds,
                       maxfun=budget,
                       approx_grad=True
                      )

history = []
def record_beer_score(x):
    history.append(x)
    return inverse_beer_score(x)

optimize.fmin_l_bfgs_b(record_beer_score, (8., 40.), bounds=bounds,
                       maxfun=budget,
                       approx_grad=True
                      )

plt.scatter(*list(zip(*history)), c=range(len(history)), cmap='viridis');
plt.xlim([2, 14])
plt.ylim([5, 80])
plt.xlabel('Alcohol')
plt.ylabel("Bitterness");



