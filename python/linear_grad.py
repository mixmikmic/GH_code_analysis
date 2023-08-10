# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
np.set_printoptions(precision=2)

# HIDDEN
tips = sns.load_dataset('tips')

# HIDDEN
def minimize(cost_fn, grad_cost_fn, x_vals, y_vals,
             alpha=0.0005, progress=True):
    '''
    Uses gradient descent to minimize cost_fn. Returns the minimizing value of
    theta once the cost changes less than 0.0001 between iterations.
    '''
    theta = np.array([0., 0.])
    cost = cost_fn(theta, x_vals, y_vals)
    while True:
        if progress:
            print(f'theta: {theta} | cost: {cost}')
        gradient = grad_cost_fn(theta, x_vals, y_vals)
        new_theta = theta - alpha * gradient
        new_cost = cost_fn(new_theta, x_vals, y_vals)
        
        if abs(new_cost - cost) < 0.0001:
            return new_theta
        
        theta = new_theta
        cost = new_cost

def simple_linear_model(thetas, x_vals):
    '''Returns predictions by a linear model on x_vals.'''
    return thetas[0] + thetas[1] * x_vals

def mse_cost(thetas, x_vals, y_vals):
    return np.mean((y_vals - simple_linear_model(thetas, x_vals)) ** 2)

def grad_mse_cost(thetas, x_vals, y_vals):
    n = len(x_vals)
    grad_0 = y_vals - simple_linear_model(thetas, x_vals)
    grad_1 = (y_vals - simple_linear_model(thetas, x_vals)) * x_vals
    return -2 / n * np.array([np.sum(grad_0), np.sum(grad_1)])

# HIDDEN
thetas = np.array([1, 1])
x_vals = np.array([3, 4])
y_vals = np.array([4, 5])
assert np.allclose(grad_mse_cost(thetas, x_vals, y_vals), [0, 0])

get_ipython().run_cell_magic('time', '', "\nthetas = minimize(mse_cost, grad_mse_cost, tips['total_bill'], tips['tip'])")

# HIDDEN
x_vals = np.array([0, 55])
sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False)
plt.plot(x_vals, simple_linear_model(thetas, x_vals), c='goldenrod')
plt.title('Tip amount vs. Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount');

