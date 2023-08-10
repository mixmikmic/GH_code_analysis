# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
# import ipywidgets as widgets
# from ipywidgets import interact, interactive, fixed, interact_manual
# import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8

# HIDDEN
tips = sns.load_dataset('tips')
tips['pcttip'] = tips['tip'] / tips['total_bill'] * 100

# HIDDEN
def mse_cost(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def abs_cost(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))

def quartic_cost(theta, y_vals):
    return np.mean(1/5000 * (y_vals - theta + 12) * (y_vals - theta + 23)
                   * (y_vals - theta - 14) * (y_vals - theta - 15) + 7)

def grad_quartic_cost(theta, y_vals):
#     return -2 * np.mean(y_vals - theta)
    return -1/2500 * (2 *(y_vals - theta)**3 + 9*(y_vals - theta)**2
                      - 529*(y_vals - theta) - 327)

def plot_cost(y_vals, xlim, cost_fn):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    costs = [cost_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(5, 3))
    plt.plot(thetas, costs, zorder=1)
    plt.xlim(*xlim)
    plt.title(cost_fn.__name__)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    
def plot_theta_on_cost(y_vals, theta, cost_fn, **kwargs):
    cost = cost_fn(theta, y_vals)
    default_args = dict(label=r'$ \theta $', zorder=2,
                        s=200, c=sns.xkcd_rgb['green'])
    plt.scatter([theta], [cost], **{**default_args, **kwargs})
    
def plot_connected_thetas(y_vals, theta_1, theta_2, cost_fn, **kwargs):
    plot_theta_on_cost(y_vals, theta_1, cost_fn)
    plot_theta_on_cost(y_vals, theta_2, cost_fn)
    cost_1 = cost_fn(theta_1, y_vals)
    cost_2 = cost_fn(theta_2, y_vals)
    plt.plot([theta_1, theta_2], [cost_1, cost_2])

# HIDDEN
def plot_one_gd_iter(y_vals, theta, cost_fn, grad_cost, alpha=2.5):
    new_theta = theta - alpha * grad_cost(theta, y_vals)
    plot_cost(pts, (-23, 25), cost_fn)
    plot_theta_on_cost(pts, theta, cost_fn, c='none',
                       edgecolor=sns.xkcd_rgb['green'], linewidth=2)
    plot_theta_on_cost(pts, new_theta, cost_fn)
    print(f'old theta: {theta}')
    print(f'new theta: {new_theta[0]}')

# HIDDEN
pts = np.array([0])
plot_cost(pts, (-23, 25), quartic_cost)
plot_theta_on_cost(pts, -21, quartic_cost)

# HIDDEN
plot_one_gd_iter(pts, -21, quartic_cost, grad_quartic_cost)

# HIDDEN
plot_one_gd_iter(pts, -9.9, quartic_cost, grad_quartic_cost)

# HIDDEN
plot_one_gd_iter(pts, -12.6, quartic_cost, grad_quartic_cost)

# HIDDEN
plot_one_gd_iter(pts, -14.2, quartic_cost, grad_quartic_cost)

# HIDDEN
pts = np.array([-2, -1, 1])
plot_cost(pts, (-5, 5), mse_cost)

# HIDDEN
pts = np.array([-1, 1])
plot_cost(pts, (-5, 5), abs_cost)

# HIDDEN
pts = np.array([0])
plot_cost(pts, (-23, 25), quartic_cost)
plot_connected_thetas(pts, -12, 12, quartic_cost)

# HIDDEN
pts = np.array([0])
plot_cost(pts, (-23, 25), mse_cost)
plot_connected_thetas(pts, -12, 12, mse_cost)

