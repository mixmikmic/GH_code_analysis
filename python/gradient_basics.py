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
def mse_cost(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)

def points_and_cost(y_vals, xlim, cost_fn):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    costs = [cost_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(*xlim)
    plt.title('Points')
    plt.xlabel('Tip Percent')
    
    ax = plt.subplot(122)
    plt.plot(thetas, costs)
    plt.xlim(*xlim)
    plt.title(cost_fn.__name__)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    plt.legend()

# HIDDEN
pts = np.array([12, 13, 15, 16, 17])
points_and_cost(pts, (11, 18), mse_cost)

def simple_minimize(cost_fn, dataset, thetas):
    '''
    Returns the value of theta in thetas that produces the least cost
    on a given dataset.
    '''
    costs = [cost_fn(theta, dataset) for theta in thetas]
    return thetas[np.argmin(costs)]

def mse_cost(theta, dataset):
    return np.mean((dataset - theta) ** 2)

dataset = np.array([12, 13, 15, 16, 17])
thetas = np.arange(12, 18, 0.1)

simple_minimize(mse_cost, dataset, thetas)

# Compute the minimizing theta using the analytical formula
np.mean(dataset)

def huber_cost(theta, dataset, alpha = 1):
    d = np.abs(theta - dataset)
    return np.mean(
        np.where(d < alpha,
                 (theta - dataset)**2 / 2.0,
                 alpha * (d - alpha / 2.0))
    )

# HIDDEN
points_and_cost(pts, (11, 18), huber_cost)

simple_minimize(huber_cost, dataset, thetas)

tips = sns.load_dataset('tips')
tips['pcttip'] = tips['tip'] / tips['total_bill'] * 100
tips.head()

# HIDDEN
points_and_cost(tips['pcttip'], (11, 20), huber_cost)

simple_minimize(huber_cost, tips['pcttip'], thetas)

print(f"          MSE cost: theta = {tips['pcttip'].mean():.2f}")
print(f"Mean Absolute cost: theta = {tips['pcttip'].median():.2f}")
print(f"        Huber cost: theta = 15.50")

sns.distplot(tips['pcttip'], bins=50);

