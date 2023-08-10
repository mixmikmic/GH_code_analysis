import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import GPy # import GPy package
np.random.seed(12345)
GPy.plotting.change_plotting_library('plotly')

# Define dataset 
N = 100
k1 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,10,10,0.1,0.1]), ARD=True)
k2 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[10,0.1,10,0.1,10]), ARD=True)
k3 = GPy.kern.RBF(5, variance=1, lengthscale=1./np.random.dirichlet(np.r_[0.1,0.1,10,10,10]), ARD=True)
X = np.random.normal(0, 1, (N, 5))
A = np.random.multivariate_normal(np.zeros(N), k1.K(X), 10).T
B = np.random.multivariate_normal(np.zeros(N), k2.K(X), 10).T
C = np.random.multivariate_normal(np.zeros(N), k3.K(X), 10).T

Y = np.vstack((A,B,C))
labels = np.hstack((np.zeros(A.shape[0]), np.ones(B.shape[0]), np.ones(C.shape[0])*2))

input_dim = 2 # How many latent dimensions to use
kernel = GPy.kern.RBF(input_dim, 1, ARD=True) 

Q = input_dim
m_gplvm = GPy.models.GPLVM(Y, Q, kernel=GPy.kern.RBF(Q))
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.
#m2.likelihood.variance.fix(.1)
m_gplvm

m_gplvm.optimize(messages=1, max_iters=5e4)

figure = GPy.plotting.plotting_library().figure(1, 2, 
                        shared_yaxes=True,
                        shared_xaxes=True,
                        subplot_titles=('Latent Space', 
                                        'Magnification',
                                        )
                            )

canvas = m_gplvm.plot_latent(labels=labels, figure=figure, col=(1), legend=False)
canvas = m_gplvm.plot_magnification(labels=labels, figure=figure, col=(2), legend=False)

GPy.plotting.show(canvas, filename='wishart_metric_notebook')

figure = GPy.plotting.plotting_library().figure(1, 3, 
                        shared_yaxes=True,
                        shared_xaxes=True,
                        subplot_titles=('Full Magnification', 
                                        'Magnification Mean',
                                        'Magnification Variance'
                                        )
                            )

canvas = m_gplvm.plot_magnification(figure=figure,
                      covariance=True, 
                      updates=False,
                      labels = labels,
                      col=1,
                     )

canvas = m_gplvm.plot_magnification(figure=figure,
                      covariance=False, 
                      updates=False,
                      labels = labels,
                      col=2,
                     )

canvas = m_gplvm.plot_magnification(figure=figure,
                      mean=False,
                      covariance=True, 
                      updates=False,
                      labels = labels,
                      col=3,
                     )

GPy.plotting.show(canvas, filename='wishart_metric_notebook_mean_variance')

GPy.plotting.change_plotting_library('matplotlib')
m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(Y, input_dim, num_inducing=30, missing_data=True)

m.optimize(messages=1, max_iters=5e3)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Latent Space')
v1 = m.plot_latent(labels=labels, ax=axes[0], updates=False)
axes[1].set_title('Magnification')
v2 = m.plot_magnification(labels=labels, ax=axes[1], updates=False, resolution=120)

m_stick = GPy.examples.dimensionality_reduction.stick_bgplvm(plot=False)

m_stick.kern.plot_ARD()

fig, axes = plt.subplots(1, 2, figsize=(15,5))
wi = [0,3]
axes[0].set_title('Latent Space')
v1 = m_stick.plot_latent(labels=None, ax=axes[0], updates=False, which_indices=wi)
axes[1].set_title('Magnification')
v2 = m_stick.plot_magnification(labels=None, ax=axes[1], updates=False, resolution=120, which_indices=wi)

m_oil = GPy.examples.dimensionality_reduction.bgplvm_oil(N=300, plot=False)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title('Latent Space')
v1 = m_oil.plot_latent(labels=m_oil.data_labels, ax=axes[0], updates=False)
axes[1].set_title('Magnification')
v2 = m_oil.plot_magnification(labels=m_oil.data_labels, ax=axes[1], updates=False, resolution=120)

