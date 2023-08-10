#import necessary modules, set up the plotting
import numpy as np
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt
import GPy

m = GPy.examples.regression.sparse_GP_regression_1D(plot=False, optimize=False)

m

m.rbf

m.inducing_inputs

m.inducing_inputs[0] = 1

m.rbf.lengthscale = 0.2
print m

print m['.*var']
#print "variances as a np.array:", m['.*var'].values()
#print "np.array of rbf matches: ", m['.*rbf'].values()

m['.*var'] = 2.
print m
m['.*var'] = [2., 3.]
print m

print m['']

new_params = np.r_[[-4,-2,0,2,4], [.1,2], [.7]]
print new_params

m[:] = new_params
print m     

m.inducing_inputs[2:, 0] = [1,3,5]
print m.inducing_inputs

precision = 1./m.Gaussian_noise.variance
print precision

print "all gradients of the model:\n", m.gradient
print "\n gradients of the rbf kernel:\n", m.rbf.gradient

m.optimize()
print m.gradient

m.rbf.variance.unconstrain()
print m

m.unconstrain()
print m

m.inducing_inputs[0].fix()
m.rbf.constrain_positive()
print m
m.unfix()
print m

m.Gaussian_noise.constrain_positive()
m.rbf.constrain_positive()
m.optimize()

fig = m.plot()

GPy.plotting.change_plotting_library('plotly')
fig = m.plot(plot_density=True)
GPy.plotting.show(fig, filename='gpy_sparse_gp_example')



