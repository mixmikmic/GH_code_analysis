get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')
from lae_regression import l1_fit
from lae_regression.plot_helper import plot
import numpy as np  # for convenience

# Trivial example usage
# Independent variable values.
U = ([1], [2])
# Dependent variable values
v = (0, 1)
# Perform the regression.
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

plot(result)

# Less trivial example usage
# Independent variable values
U = ([1], [2], [3])
# Dependent variable values
v = (0, 0, 1)
result = l1_fit(U, v)
plot(result)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# A trivial example with 2 predictor variables x + y - 2
U = ([1, 0], [2, 1], [3, 0])
v = (-1, 1, 1)
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# A non trivial example with 2 predictor variables x + y - 2 and one outlier
U = ([1, 1], [1, -1], [-1, -1], [-1, 1], [0, 0])
v = (0, -2, -4, -2, -1.3)
result = l1_fit(U, v)
result["m"], result["k"], result["residuals"], result["samples"], result["dimensions"]

# nearly linear with an outlier
n = 10
x = np.linspace(0,1,n)
small_error = (np.random.randn(n)-0.5) * 0.1
y = (0.5 * x - 1) + small_error
y[5] = 5

U = x.reshape((n, 1))
v = y
result = l1_fit(U, v)
plot(result)

# a random 1d example
n = 20
x = np.random.randn(n)
y = np.random.randn(n)
U = x.reshape((n, 1))
v = y
result = l1_fit(U, v)
plot(result)



