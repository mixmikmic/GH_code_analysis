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

from ipywidgets import interact

from skopt import Optimizer
from skopt.space import Space
from skopt.utils import cook_estimator
from utils import plot_optimizer


x = np.linspace(-2, 2, 400).reshape(-1, 1)
noise_level = 0.1

# Our 1D toy problem, this is the function we are trying to
# minimize
def objective(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

# setup the dimensions of the space, the surrogate model to use and plug it all together
# inside the optimizer
space = Space([(-2.0, 2.0)])

# use our knowledge of the size of the noise
#gp = cook_estimator("GP", space, noise=0.1**2)
# or not use it
gp = cook_estimator("GP", space, noise="gaussian")

@interact(xi=(0.01, 1.), kappa=(0., 3.), n_iterations=(3, 20), acq_name=['EI', 'LCB'])
def run(xi=0.01, kappa=1.96, n_iterations=4, acq_name='EI'):
    np.random.seed(123+1+1+1)

    if acq_name == "EI":
        acq_func_kwargs = {'xi': xi}
    elif acq_name == 'LCB':
        acq_func_kwargs = {'kappa': kappa}

    opt = Optimizer(space, gp, acq_func=acq_name, n_initial_points=3,
                    random_state=3,
                    #acq_optimizer='sampling',
                    acq_func_kwargs=acq_func_kwargs)

    for _ in range(n_iterations):
        suggested = opt.ask()
        y = objective(suggested)
        opt.tell(suggested, y)   
        
    plot_optimizer(opt, x, acq_name=acq_name)
    plt.xlim([-2, 2])
    plt.ylim([-2, 1])
    plt.show()
# default settings, xi=0.01 more exploitation
# xi=5, more exploration

