import GPy, numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

X = np.random.uniform(0, 10, (200, 1))
f = np.sin(.3*X) + .3*np.cos(1.3*X)
f -= f.mean()
Y = f+np.random.normal(0, .1, f.shape)

plt.scatter(X, Y)

m = GPy.models.GPRegression(X, Y)
m

m.rbf.lengthscale = 1.5
m

# Type your code here

m.optimize(messages=1)

_ = m.plot()

# You can use different kernels to use on the data.
# Try out three different kernels and plot the result after optimizing the GP:
# See kernels using GPy.kern.<tab>

# Type your code here

# Type your code here

# Type your code here

get_ipython().magic('pinfo GPy.core.SparseGP')

get_ipython().magic('pinfo GPy.core.SVGP')

#Type your code here

