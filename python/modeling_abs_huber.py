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

def abs_cost(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))

# HIDDEN
def compare_mse_abs(thetas, y_vals, xlims, figsize=(10, 7), cols=3):
    if not isinstance(y_vals, np.ndarray):
        y_vals = np.array(y_vals)
    rows = int(np.ceil(len(thetas) / cols))
    plt.figure(figsize=figsize)
    for i, theta in enumerate(thetas):
        ax = plt.subplot(rows, cols, i + 1)
        sns.rugplot(y_vals, height=0.1, ax=ax)
        plt.axvline(theta, linestyle='--',
                    label=rf'$ \theta = {theta} $')
        plt.title(f'MSE Cost = {mse_cost(theta, y_vals):.2f}\n'
                  f'Mean Abs Cost = {abs_cost(theta, y_vals):.2f}')
        plt.xlim(*xlims)
        plt.yticks([])
        plt.legend()
    plt.tight_layout()

# HIDDEN
compare_mse_abs(thetas=[11, 12, 13, 14, 15, 16],
                y_vals=[14], xlims=(10, 17))

# HIDDEN
compare_mse_abs(thetas=[12, 13, 14, 15, 16, 17],
                y_vals=[12.1, 12.8, 14.9, 16.3, 17.2],
                xlims=(11, 18))

# HIDDEN
thetas = np.array([12, 13, 14, 15, 16, 17])
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mse_costs = [mse_cost(theta, y_vals) for theta in thetas]
abs_costs = [abs_cost(theta, y_vals) for theta in thetas]

plt.scatter(thetas, mse_costs, label='MSE Cost')
plt.scatter(thetas, abs_costs, label='Abs Cost')
plt.title(r'Cost vs. $ \theta $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Cost')
plt.legend();

# HIDDEN
thetas = np.arange(12, 17.1, 0.05)
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mse_costs = [mse_cost(theta, y_vals) for theta in thetas]
abs_costs = [abs_cost(theta, y_vals) for theta in thetas]

plt.plot(thetas, mse_costs, label='MSE Cost')
plt.plot(thetas, abs_costs, label='Abs Cost')
plt.title(r'Cost vs. $ \theta $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Cost')
plt.legend();

# HIDDEN
thetas = np.arange(12, 17.1, 0.05)
y_vals = np.array([12.1, 12.8, 14.9, 16.3, 17.2])
mse_costs = [mse_cost(theta, y_vals) for theta in thetas]
abs_costs = [abs_cost(theta, y_vals) for theta in thetas]

plt.plot(thetas, mse_costs, label='MSE Cost')
plt.plot(thetas, abs_costs, label='Abs Cost')
plt.axvline(np.mean(y_vals), c=sns.color_palette()[0], linestyle='--',
            alpha=0.7, label='Minima of MSE cost')
plt.axvline(np.median(y_vals), c=sns.color_palette()[1], linestyle='--',
            alpha=0.7, label='Minima of abs cost')


plt.title(r'Cost vs. $ \theta $ when $ y = [ 12.1, 12.8, 14.9, 16.3, 17.2 ] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Cost')
plt.ylim(1.5, 5)
plt.legend();

# HIDDEN
def compare_mse_abs_curves(y3=14):
    thetas = np.arange(11.5, 26.5, 0.1)
    y_vals = np.array([12, 13, y3])
    
    mse_costs = [mse_cost(theta, y_vals) for theta in thetas]
    abs_costs = [abs_cost(theta, y_vals) for theta in thetas]
    mse_abs_diff = min(mse_costs) - min(abs_costs)
    mse_costs = [cost - mse_abs_diff for cost in mse_costs]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(11.5, 26.5)
    plt.xlabel('Points')
    
    ax = plt.subplot(122)
    plt.plot(thetas, mse_costs, label='MSE Cost')
    plt.plot(thetas, abs_costs, label='Abs Cost')
    plt.xlim(11.5, 26.5)
    plt.ylim(min(abs_costs) - 1, min(abs_costs) + 10)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    plt.legend()

# HIDDEN
interact(compare_mse_abs_curves, y3=(14, 25));

# HIDDEN
compare_mse_abs_curves(y3=14)

# HIDDEN
compare_mse_abs_curves(y3=25)

# HIDDEN
def points_and_cost(y_vals, xlim, cost_fn=abs_cost):
    thetas = np.arange(xlim[0], xlim[1] + 0.01, 0.05)
    abs_costs = [cost_fn(theta, y_vals) for theta in thetas]
    
    plt.figure(figsize=(9, 2))
    
    ax = plt.subplot(121)
    sns.rugplot(y_vals, height=0.3, ax=ax)
    plt.xlim(*xlim)
    plt.xlabel('Points')
    
    ax = plt.subplot(122)
    plt.plot(thetas, abs_costs)
    plt.xlim(*xlim)
    plt.xlabel(r'$ \theta $')
    plt.ylabel('Cost')
    plt.legend()
points_and_cost(np.array([10, 11, 12, 14, 15]), (9, 16))

# HIDDEN
points_and_cost(np.array([10, 11, 14, 15]), (9, 16))

# HIDDEN
points_and_cost(np.array([10, 11, 14, 15]), (9, 16), mse_cost)

# HIDDEN
def huber_loss(est, y_obs, alpha = 1):
    d = np.abs(est - y_obs)
    return np.where(d < alpha, 
                    (est - y_obs)**2 / 2.0,
                    alpha * (d - alpha / 2.0))

thetas = np.linspace(0, 50, 200)
loss = huber_loss(thetas, np.array([14]), alpha=5)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\theta$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')

# HIDDEN
loss = huber_loss(thetas, np.array([14]), alpha=1)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\theta$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')

# HIDDEN
loss = huber_loss(thetas, np.array([14]), alpha=10)
plt.plot(thetas, loss, label="Huber Loss")
plt.vlines(np.array([14]), -20, -5,colors="r", label="Observation")
plt.xlabel(r"Choice for $\theta$")
plt.ylabel(r"Loss")
plt.legend()
plt.savefig('huber_loss.pdf')

