get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
plt.style.use("ggplot")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hard_sigmoid(x):
    return min(1, max(0, x))

def hard_concrete(loc, temp, gamma, zeta):
    u = np.random.random()
    s = sigmoid((np.log(u) - np.log(1 - u) + loc) / temp)
    s = s * (zeta - gamma) + gamma
    return hard_sigmoid(s)

def plot_hard_concreate(loc, temp, gamma, zeta, num=10_000, bins=100, **kwargs):
    plt.hist([hard_concrete(loc, temp, gamma, zeta) for _ in range(num)], bins=bins, density=True, **kwargs)

def interactive_hard_concrete(loc, temp):
    plot_hard_concreate(loc, temp, gamma=-0.1, zeta=1.1)

interact(interactive_hard_concrete, loc=(-0.5, 0.5), temp=(0.01, 1));



