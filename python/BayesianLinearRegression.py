import numpy as np, GPy, itertools, pandas as pd, seaborn as sns, sklearn.cross_validation as cv
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

n = 300

# define function to add the intercept on a design matrix:
def _add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# define linear design matrix with an intercept entry as first dimension
def linear(*args, **kwargs):
    def inner(X):
        return _add_intercept(X)
    return inner

# define convenience function for plotting data
def plot_data(X, y, cmap=None, ax=None, *args, **kwargs):
    if cmap is None:
        cmap = cm.coolwarm
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    if not 'marker' in kwargs:
        kwargs['marker'] = 'x'
    if not 'linewidths' in kwargs:
        kwargs['linewidths'] = 1.5
    ax.scatter(*X.T, zs=y[:, 0], c=y[:, 0], cmap=cmap, **kwargs)
    return ax

np.random.seed(12345)
X = np.random.uniform(-5, 5, (n, 2))
Phi = linear()(X)
w_train = np.random.normal(0, 1, (3, 1))
print '\n'.join(['training w_{} = {}'.format(i, float(w_train[i])) for i in range(w_train.shape[0])])
y = Phi.dot(w_train) + np.random.normal(0, .4, (n, 1))

_ = plot_data(X, y)

# Prediction is the simple multiplication with the right parameters w
def predict_basis(Phi, w):
    return Phi.dot(w)

# convenience function to fit a dataset with a basis function `basis`
def fit_linear(Phi, y):
    return np.linalg.solve(Phi.T.dot(Phi), Phi.T.dot(y)), 0

def error(y_true, y_pred):
    StdE = np.sqrt(np.sum(np.square(y_true-y_pred))/(y_true.shape[0]*(y_true.shape[0]-1)))
    RMSE = np.sqrt(np.mean(np.square(y_true-y_pred)))
    return StdE, RMSE

basis = linear()
Phi = basis(X)
w_linear, _ = fit_linear(Phi, y)
print '\n'.join(['w_{} = {}'.format(i, float(w_linear[i])) for i in range(w_linear.shape[0])])

def _range_X(X, perc=.0):
    xmi, xma = X.min(0), X.max(0)
    ra = xma-xmi
    return xmi, xma, perc*ra

def plot_grid(X, resolution=30):
    xmi, xma, ra = _range_X(X)
    grid = np.mgrid[1:1:1j, 
                    xmi[0]-ra[0]:xma[0]+ra[0]:complex(resolution), 
                    xmi[1]-ra[1]:xma[1]+ra[1]:complex(resolution)]
    plot_X = plot_x, plot_y = grid[1:, 0] # unpack the right dimensions, have a look inside, if you are interested
    return plot_x, plot_y, plot_X.reshape(2, -1).T

def plot_predict(X, y, w, basis, predict, cmap=None, resolution=30, ax=None, alpha=.7):
    if cmap is None:
        cmap = cm.coolwarm
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    plot_x, plot_y, plot_X = plot_grid(X, resolution=resolution)
    Phi_pred = basis(plot_X)
    y_pred = predict(Phi_pred, w)
    plot_z = y_pred[:, 0].reshape(plot_x.shape)
    _ = ax.plot_surface(plot_x, plot_y, plot_z, alpha=alpha, cmap=cmap, antialiased=True, rstride=1, cstride=1, linewidth=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_predict(X, y, w_linear, linear(), predict_basis, ax=ax)
plot_data(X, y, ax=ax)

def polynomial(degree):
    def inner(X):
        if degree == 0:
            return np.ones((X.shape[0], 1))
        return _add_intercept(np.concatenate([X**(i+1) for i in range(degree)], axis=1))
    return inner

def exponentiated_quadratic(degree, lengthscale, domain):
    """
    degree: number of basis functions to use
    lengthscale: the scale of the basis functions
    domain: input X, so that the basis functions will span the domain of X
    """
    xmi, xma = domain.min(0), domain.max(0)
    ra = xma-xmi
    def inner(X):
        if degree == 0:
            return np.ones((X.shape[0], 1))
        if degree == 1:
            return _add_intercept(np.exp(-.5*((X)/lengthscale)**2))
        return _add_intercept(np.concatenate([
                    np.exp(-.5*(((X-xmi)-(float(i)*(ra/(degree-1))))/lengthscale)**2) for i in range(degree)], axis=1))
    return inner

def plot_basis(X, basis, w=None, plot_intercept=False, resolution=25, ax=None):
    """
    plot the `basis` in a grid with `resolution` squares around the input X.
    *args, **kwargs are passed to the basis function.
    Only in one dimension, as otherwise it gets to cluttered.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    plot_x, plot_y, plot_X = plot_grid(X, resolution=resolution)
    Phi_pred = basis(plot_X)
    if w is None:
        w = np.ones((Phi_pred.shape[1], 1))
    Phi_pred *= w.T
        
    if plot_intercept: to_plot = np.r_[0, 1:Phi_pred.shape[1]:2]
    else: to_plot = np.r_[1:Phi_pred.shape[1]:2]

    for i, z in enumerate(to_plot):
        c = cm.cool(float(i)/(2*Phi_pred.shape[1]))
        plot_z = Phi_pred[:, z]
        _ = ax.plot_surface(plot_x, plot_y, plot_z.reshape(resolution, resolution), alpha=.7, antialiased=True, color=c, rstride=1, cstride=1, linewidth=0)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12,4.5), subplot_kw=dict(projection='3d'))

plot_basis(X, exponentiated_quadratic(6, 4, X), plot_intercept=0, resolution=30, ax=ax1)
plot_basis(X, polynomial(2), plot_intercept=1, resolution=30, ax=ax2)

y = (2.5 + (np.sin(X-2) + np.cos(.2*X) + .1*X**2 + .2*X + .01*X**3).sum(1))[:, None]
#y = (y-y.mean())/y.std()
y += np.random.normal(0, .1, (X.shape[0], 1))

X_train, X_test, y_train, y_test = cv.train_test_split(X, y, train_size=.1)

def optimal_ax_grid(n):
    i=1;j=1
    while i*j < n:
        if i==j or j==(i+1): j+=1
        elif j==(i+2): j-=1; i+=1
    return i,j
            
def plot_fits(X_train, y_train, X_test, y_test, basis_gen, predict, degrees, fit=fit_linear, figsize=None):
    i, j = optimal_ax_grid(len(degrees))
    fig, axes = plt.subplots(i, j, figsize=figsize, subplot_kw=dict(projection='3d'))
    axes = axes.flat
    errors = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'test'], ['RMSE', 'StdE']]))
    for i, d in enumerate(degrees):
        basis = basis_gen(d)
        Phi = basis(X_train)
        w_fit, var = fit(Phi, y_train)
        StdE, RMSE = _e = error(y_train, predict(Phi, w_fit))
        Phi_test = basis(X_test)
        StdE_test, RMSE_test = _e_test= error(y_test, predict(Phi_test, w_fit))
        errors.loc[d, ['train']] = _e[::-1]
        errors.loc[d, ['test']] = _e_test[::-1]
        ax = axes[i]
        plot_predict(X_train, y_train, w_fit, basis, predict, ax=ax)
        plot_data(X_train, y_train, ax=ax, marker='o', label='train', edgecolors='k')
        plot_data(X_test, y_test, ax=ax, marker='o', label='test', edgecolors='w')
        ax.set_title('degree={}\nTrain StdE={:.4f}\nTest StdE={:.4f}'.format(d, StdE, StdE_test))
        ax.legend()
    return errors

degrees = np.arange(1,10,1)

poly_errors = plot_fits(X_train, y_train, X_test, y_test, polynomial, predict_basis, degrees=degrees, figsize=(12,10))

eq_errors = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis, degrees=degrees, figsize=(12,10))

linear_df = pd.concat((eq_errors.T, poly_errors.T), keys=['eq', 'poly']).reset_index(0)
#eq_errors.plot(kind='bar', ax=ax, label='eq')
#.plot(kind='bar', ax=ax, label='poly', xlabel='degree')
_c = linear_df.columns.tolist(); _c[0] = 'basis'; linear_df.columns=_c
#df.reset_index()
linear_df = linear_df.reset_index().set_index(['basis', 'level_0', 'level_1']).T

fig, [ax1, ax2] = fig, axes = plt.subplots(1,2,figsize=(12, 2.5), sharex=True, sharey=True)
fig, [ax3, ax4] = fig, axes = plt.subplots(1,2,figsize=(12, 2.5), sharex=True, sharey=False)

linear_df.index.name='degree'

eq_df = linear_df['eq'].xs('StdE', level=1, axis=1)
eq_df.columns.name = 'exponentiaded quadratic'
eq_df.plot(kind='line', ax=ax1)

poly_df = linear_df['poly'].xs('StdE', level=1, axis=1)
poly_df.columns.name = 'polynomial'
poly_df.plot(kind='line', ax=ax2)

train_df = linear_df.xs('train', level=1, axis=1).xs('StdE', level=1, axis=1)
train_df.columns.name = 'training'
train_df.plot(kind='line', ax=ax3)

test_df = linear_df.xs('test', level=1, axis=1).xs('StdE', level=1, axis=1)
test_df.columns.name = 'test'
test_df.plot(kind='line', ax=ax4)

ax1.set_ylabel('StdE')
ax2.set_ylabel('StdE')
ax3.set_ylabel('StdE')
ax4.set_ylabel('StdE')
ax1.axes.set_yscale('log')
ax2.axes.set_yscale('log')
ax3.axes.set_yscale('log')
ax4.axes.set_yscale('log')

# convenience function to solve the Baysian linear system using the design matrix Phi
def fit_bayesian(Phi, y, sigma=.1, alpha=20):
    Cinv = (Phi.T.dot(Phi)/sigma**2 + np.eye(Phi.shape[1])/alpha)
    return np.linalg.solve(Cinv, Phi.T.dot(y)/sigma**2), np.linalg.inv(Cinv)

basis = exponentiated_quadratic(10, 4, X)
Phi = basis(X)
w_linear = fit_linear(Phi, y)
w_bayesian, w_cov = fit_bayesian(Phi, y)

linear_error = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis,
                         degrees=np.arange(3, 10, 3), 
                         figsize=(12,3), 
                         fit=fit_linear)

bayesian_errors = pd.DataFrame(columns=['degree', 'sigma', 'split', 'RMSE', 'StdE'])
def concat_be(bayesian_errors, _be, sigma):
    _be.index.name='degree'
    _be = _be[['test', 'train']].stack(0).reset_index(1)
    _be['sigma'] = sigma
    _be = _be.rename(columns=dict(level_1='split'))
    return pd.concat((bayesian_errors, _be.reset_index()), axis=0)

sigma = .2
print 'Bayesian fit with noise variance of {}'.format(sigma)
_be = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis,
                            degrees=np.arange(3, 10, 3),
                            figsize=(12,3), 
                            fit=lambda X, y: fit_bayesian(X, y, sigma=sigma))
bayesian_errors = concat_be(bayesian_errors, _be, sigma)

sigma = .1
print 'Bayesian fit with noise variance of {}'.format(sigma)
_be = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis,
                            degrees=np.arange(3, 10, 3),
                            figsize=(12,3), 
                            fit=lambda X, y: fit_bayesian(X, y, sigma=sigma))
bayesian_errors = concat_be(bayesian_errors, _be, sigma)

sigma = .01
print 'Bayesian fit with noise variance of {}'.format(sigma)
_be = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis, 
                            degrees=np.arange(3, 10, 3),
                            figsize=(12,3), 
                            fit=lambda X, y: fit_bayesian(X, y, sigma=sigma))
bayesian_errors = concat_be(bayesian_errors, _be, sigma)

sigma = .001
print 'Bayesian fit with noise variance of {}'.format(sigma)
_be = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis, 
                            degrees=np.arange(3, 10, 3),
                            figsize=(12,3), 
                            fit=lambda X, y: fit_bayesian(X, y, sigma=sigma))
bayesian_errors = concat_be(bayesian_errors, _be, sigma)

sigma = 1e-10
print 'Bayesian fit with noise variance of {}'.format(sigma)
_be = plot_fits(X_train, y_train, X_test, y_test, lambda d: exponentiated_quadratic(d, 2, X), predict_basis, 
                            degrees=np.arange(3, 10, 3),
                            figsize=(12,3), 
                            fit=lambda X, y: fit_bayesian(X, y, sigma=sigma))
bayesian_errors = concat_be(bayesian_errors, _be, sigma)

fig, [ax1, ax2] = plt.subplots(1,2,figsize=(12, 3), sharex=True, sharey=True)
tmp = bayesian_errors[bayesian_errors['split']=='train'][['StdE', 'sigma', 'degree']].set_index(['sigma', 'degree'])
tmp = tmp.unstack(level=0)
tmp.columns = tmp.columns.droplevel()
tmp.plot(kind='bar', ax=ax1)

tmp = bayesian_errors[bayesian_errors['split']=='test'][['StdE', 'sigma', 'degree']].set_index(['sigma', 'degree'])
tmp = tmp.unstack(level=0)
tmp.columns = tmp.columns.droplevel()
tmp.plot(kind='bar', ax=ax2)
ax1.set_ylabel('StdE')

ax1.set_title('training')
ax2.set_title('test')

bayesian_errors[bayesian_errors['split']=='test'].sort(['StdE', 'RMSE']).head(3)

Phi = basis(X_train)
kernel = GPy.kern.Fixed(2, Phi.dot(Phi.T))
kernel.name = 'data'
kernel.variance.name = 'alpha'
m_max_lik = GPy.models.GPRegression(X_train, y_train, kernel)
m_max_lik.likelihood.name = 'noise'
m_max_lik.likelihood.variance.name = 'sigma'

m_max_lik.optimize()

m_max_lik

Phi = basis(X_train)
Phi_test = basis(X_test)

def predict_max_lik(Phi_test, w):
    return (m_max_lik.kern.variance) * Phi_test.dot(Phi.T.dot(m_max_lik.posterior.woodbury_vector))

fig, ax = plt.subplots(1, 1, figsize=(8,6), subplot_kw=dict(projection='3d'))
plot_predict(X_train, y_train, None, basis, predict_max_lik, ax=ax)
plot_data(X_train, y_train, ax=ax, marker='o', label='train', edgecolors='k')
plot_data(X_test, y_test, ax=ax, marker='o', label='test', edgecolors='none')

max_lik_pred = predict_max_lik(Phi_test, None)
max_likelihood_StdE, max_likelihood_RMSE = error(y_test, max_lik_pred)

mu, var = fit_bayesian(Phi, y_train, sigma=np.sqrt(m_max_lik.likelihood.variance[0]), alpha=m_max_lik.kern.variance[0])
std = np.sqrt(np.diagonal(var))

w_realizations = np.empty((mu.shape[0], 9))
for i in range(w_realizations.shape[1]):
    w_realization = np.random.multivariate_normal(mu[:, 0], var)[:, None]
    w_realizations[:, i] = w_realization[:,0]

fig, axes = plt.subplots(1, 1, figsize=(12,9), subplot_kw=dict(projection='3d'))
for i in range(9):
    ax = axes
    plot_predict(X_train, y_train, w_realizations[:, [i]], basis, predict_basis, ax=ax, alpha=.1)
plot_data(X_train, y_train, ax=ax, marker='o', label='train', edgecolors='k')
plot_data(X_test, y_test, ax=ax, marker='o', label='test', edgecolors='none')

fig, axes = plt.subplots(3, 3, figsize=(12,9), subplot_kw=dict(projection='3d'))
for i in range(w_realizations.shape[1]):
    ax = axes.flat[i]
    plot_basis(X_test, basis, w_realizations[:, i], plot_intercept=1, resolution=30, ax=ax)

k = GPy.kern.RBF(2, ARD=0) + GPy.kern.Bias(2) + GPy.kern.Linear(2, ARD=0)
m = GPy.models.GPRegression(X_train, y_train, k)
m.optimize()
y_pred_gp, y_pred_gp_var = m.predict(X_test)
gp_StdE, gp_RMSE = error(y_test, y_pred_gp)
m

fig, ax = plt.subplots(1, 1, figsize=(8,6), subplot_kw=dict(projection='3d'))
basis = lambda x: x
def predict_gp(X, w):
    return m.predict(X)[0]
plot_predict(X_train, y_train, None, basis, predict_gp, ax=ax)
plot_data(X_train, y_train, ax=ax, marker='o', label='train', edgecolors='k')
plot_data(X_test, y_test, ax=ax, marker='o', label='test', edgecolors='none')

comparison = pd.DataFrame(columns=['linear fit', 'bayesian fit', 'max likelihood'])
comparison['linear fit'] = [linear_df['eq']['test'].min(0).loc['StdE']]
comparison['bayesian fit'] = bayesian_errors[bayesian_errors.split=='test'].min(0).loc['StdE']
comparison['max likelihood'] = max_likelihood_StdE
comparison['Gaussian process'] = gp_StdE
fig, axes = plt.subplots(1, 2, figsize=(12,3), sharey=True)
comparison[['linear fit', 'bayesian fit']].plot(kind='bar', title='Crossvalidated', ax=axes[0])
comparison[['max likelihood', 'Gaussian process']].plot(kind='bar', title='Data Driven', ax=axes[1])
plt.ylabel('StdE')
_ = plt.xticks([])

