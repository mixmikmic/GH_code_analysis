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

from skopt import plots

from skopt import gp_minimize

res = gp_minimize(inverse_beer_score, bounds,
                  n_calls=budget,
                  n_random_starts=20,
                  random_state=2)

print("best recipe (alcohol, bitterness): %.2f, %.2f" % (res.x[0], res.x[1]))
print("with a score of: %.4f" % res.fun)

plots.plot_convergence(res);

plt.scatter(*zip(*res.x_iters), c=range(len(res.x_iters)))
plt.xlabel('Alcohol')
plt.ylabel("Bitterness")
plt.colorbar();







h = 0.02
xx, yy = np.meshgrid(np.arange(2, 14, h),
                     np.arange(5, 80, h))
Z = inverse_beer_score(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='viridis_r')
plt.colorbar();

plt.scatter(*res.x, marker="*", color='r', s=500)
plt.ylabel('bitterness')
plt.xlabel('alcohol');

