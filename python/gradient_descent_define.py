# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

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

def grad_mse_cost(theta, y_vals):
    return -2 * np.mean(y_vals - theta)

def plot_cost(y_vals, xlim, cost_fn):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    costs = [cost_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(5, 3))
    plt.plot(thetas, costs, zorder=1)
    plt.xlim(*xlim)
    plt.title(cost_fn.__name__)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    plt.legend()
    
def plot_theta_on_cost(y_vals, theta, cost_fn, **kwargs):
    cost = cost_fn(theta, y_vals)
    default_args = dict(label=r'$ \theta $', zorder=2,
                        s=200, c=sns.xkcd_rgb['green'])
    plt.scatter([theta], [cost], **{**default_args, **kwargs})

def plot_tangent_on_cost(y_vals, theta, cost_fn, eps=1e-6):
    slope = ((cost_fn(theta + eps, y_vals) - cost_fn(theta - eps, y_vals))
             / (2 * eps))
    xs = np.arange(theta - 1, theta + 1, 0.05)
    ys = cost_fn(theta, y_vals) + slope * (xs - theta)
    plt.plot(xs, ys, zorder=3, c=sns.xkcd_rgb['green'], linestyle='--')

# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
plot_cost(pts, (11, 18), mse_cost)
plot_theta_on_cost(pts, 12, mse_cost)

# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
plot_cost(pts, (11, 18), mse_cost)
plot_tangent_on_cost(pts, 12, mse_cost)

# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
plot_cost(pts, (11, 18), mse_cost)
plot_tangent_on_cost(pts, 16.5, mse_cost)

# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
plot_cost(pts, (11, 18), mse_cost)
plot_theta_on_cost(pts, 12, mse_cost, c='none',
                   edgecolor=sns.xkcd_rgb['green'], linewidth=2)
plot_theta_on_cost(pts, 17.2, mse_cost)

# HIDDEN
def plot_one_gd_iter(y_vals, theta, cost_fn, grad_cost, alpha=0.3):
    new_theta = theta - alpha * grad_cost(theta, y_vals)
    plot_cost(pts, (11, 18), cost_fn)
    plot_theta_on_cost(pts, theta, cost_fn, c='none',
                       edgecolor=sns.xkcd_rgb['green'], linewidth=2)
    plot_theta_on_cost(pts, new_theta, cost_fn)
    print(f'old theta: {theta}')
    print(f'new theta: {new_theta}')

# HIDDEN
plot_one_gd_iter(pts, 12, mse_cost, grad_mse_cost)

# HIDDEN
plot_one_gd_iter(pts, 13.56, mse_cost, grad_mse_cost)

# HIDDEN
plot_one_gd_iter(pts, 14.18, mse_cost, grad_mse_cost)

# HIDDEN
plot_one_gd_iter(pts, 14.432, mse_cost, grad_mse_cost)

def minimize(cost_fn, grad_cost_fn, dataset, alpha=0.2, progress=True):
    '''
    Uses gradient descent to minimize cost_fn. Returns the minimizing value of
    theta once theta changes less than 0.001 between iterations.
    '''
    theta = 0
    while True:
        if progress:
            print(f'theta: {theta:.2f} | cost: {cost_fn(theta, dataset):.2f}')
        gradient = grad_cost_fn(theta, dataset)
        new_theta = theta - alpha * gradient
        
        if abs(new_theta - theta) < 0.001:
            return new_theta
        
        theta = new_theta

def mse_cost(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def grad_mse_cost(theta, y_vals):
    return -2 * np.mean(y_vals - theta)

get_ipython().run_cell_magic('time', '', "theta = minimize(mse_cost, grad_mse_cost, np.array([12, 13, 15, 16, 17]))\nprint(f'Minimizing theta: {theta}')\nprint()")

np.mean([12, 13, 15, 16, 17])

def huber_cost(theta, dataset, delta = 1):
    d = np.abs(theta - dataset)
    return np.mean(
        np.where(d <= delta,
                 (theta - dataset)**2 / 2.0,
                 delta * (d - delta / 2.0))
    )

def grad_huber_cost(theta, dataset, delta = 1):
    d = np.abs(theta - dataset)
    return np.mean(
        np.where(d <= delta,
                 -(dataset - theta),
                 -delta * np.sign(dataset - theta))
    )

get_ipython().run_cell_magic('time', '', "theta = minimize(huber_cost, grad_huber_cost, tips['pcttip'], progress=False)\nprint(f'Minimizing theta: {theta}')\nprint()")

