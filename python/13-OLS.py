# Imports
get_ipython().magic('matplotlib inline')

import random
import numpy as np
import matplotlib.pyplot as plt

# Create some data
# x is an evenly space array of integers
x = np.arange(0, 6)

# y is some data with underlying relationship y = (theta) * x
#  For this example, the true relation of the data is y = 2x
true_rel = 2
y = true_rel * x

# Add some noise to the y dimension
noise = np.random.normal(0, 0.5, len(x))
y = y + noise

# Plot the data
f = plt.figure()
plt.plot(x, y, '.');

# Reshape data to play nice with numpy
x = np.reshape(x, [len(x), 1])
y = np.reshape(y, [len(y), 1])

# Fit the (Ordinary) Least Squares best fit line using numpy
#  This gives us a fit value (theta), and residuals (how much error we have in this fit)
theta, residuals, _, _ = np.linalg.lstsq(x, y)

# Pull out theta value from array
theta = theta[0][0]

# Check what the OLS derived solution for theta is:
print(theta)

# Check how good our OLS solution is
print('The true relationship between y & x is: \t', true_rel)
print('OLS calculated relationship between y & x is: \t', theta)

# Check what the residuals are
residuals[0]

# Plot the raw data, with the true underlying relationship, and the OLS fit
fig, ax = plt.subplots(1)
ax.plot(x, y, 'x', markersize=10, label='Data')
ax.plot(x, 2*x, '--b', alpha=0.4, label='OLS Fit')
ax.plot(x, theta*x, label='True Fit')
ax.legend();

# With our model, we can predict the value of a new 'x' datapoint
new_x = 2.5
pred_y = theta * new_x
print('The prediction for a new x of {} is {:1.3f}'.format(new_x, pred_y))

ax.plot(new_x, pred_y, 'or')
fig

# We can also see what the model would predict for all the points we did observe
preds = theta * x

# Residuals are the just the sum of squares between the model fit and the observed data points
# Re-calculate the residuals 'by hand'
error = np.sum(np.subtract(preds, y) ** 2)

# Check that our residuals calculation matches the scipy implementation
print('Error from :', residuals[0])
print('Error from :', error)

