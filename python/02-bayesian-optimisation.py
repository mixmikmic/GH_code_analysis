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
from utils import plot_optimizer


x = np.linspace(-2, 2, 400).reshape(-1, 1)
noise_level = 0.1

# Our 1D toy problem, this is the function we are trying to
# minimize
def objective(x, noise_level=noise_level):
    return (np.sin(5 * x[0]) *
            (1 - np.tanh(x[0] ** 2)) +
            np.random.randn() * noise_level)

@interact(n_iter=(3, 22))
def show_optimizer(n_iter=3):
    np.random.seed(123)

    # setup the dimensions of the space, the 
    # surrogate model to use and plug it all
    # together inside the optimizer
    space = Space([(-2.0, 2.0)])

    opt = Optimizer(space,
                    n_initial_points=3,
                    acq_func="EI",
                    acq_func_kwargs={'xi': 0.01})
    for _ in range(n_iter):
        suggested = opt.ask()
        y = objective(suggested)
        opt.tell(suggested, y)
        
    plot_optimizer(opt, x)
    plt.show()



