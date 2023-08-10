get_ipython().magic('matplotlib inline')
get_ipython().magic('config IPython.matplotlib.backend = "retina"')

# Start by importing the specific code for this project.
import gp

# And then the standard numerical Python modules.
import emcee
import numpy as np
import matplotlib.pyplot as plt

# Finally set up the plotting to look a little nicer.
from matplotlib import rcParams
rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100

t = np.linspace(-5, 5, 5000)
model = gp.SillyTransitModel(log_depth=np.log(200.0), log_duration=np.log(0.5), time=0.0)
plt.plot(t, model.get_value(t))
plt.xlabel("time [days]")
plt.ylabel("flux [ppm]");

print("parameter_dict:")
print(model.get_parameter_dict())

print("\nparameter_vector:")
print(model.get_parameter_vector())

print("\nyou can freeze and thaw parameters...")
print("the depth, for example, is now fixed to the current value:")
model.freeze_parameter("log_depth")
print(model.get_parameter_dict())

print("\nand 'thawed':")
model.thaw_parameter("log_depth")
print(model.get_parameter_dict())

t, y, yerr = gp.data.generate_dataset()
plt.errorbar(t, y, yerr=yerr, fmt=".k")
plt.xlabel("time [days]")
plt.ylabel("flux [ppm]");

def simple_log_prob(params, model, t, y, yerr):
    model.set_parameter_vector(params)
    resid = y - model.get_value(t)
    return -0.5 * np.sum((resid/yerr)**2)

ndim = 3
nwalkers = 16
pos = 1e-4*np.random.randn(nwalkers, ndim)
pos[:, 0] += 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, simple_log_prob,
                                args=(model, t, y, yerr))
pos, _, _ = sampler.run_mcmc(pos, 200)
sampler.reset()
sampler.run_mcmc(pos, 2000);

gp.corner(sampler.flatchain, truths=gp.data.true_parameters,
          labels=["ln(depth)", "ln(duration)", "transit time"]);

def log_like(r, K):
    """
    The multivariate Gaussian ln-likelihood (up to a constant) for the
    vector ``r`` given a covariance matrix ``K``.
    
    :param r: ``(N,)``   The residual vector with ``N`` points.
    :param K: ``(N, N)`` The square (``N x N``) covariance matrix.
    
    :returns lnlike: ``float`` The Gaussian ln-likelihood. 
    
    """
    # Erase the following line and implement the Gaussian process
    # ln-likelihood here.
    pass

gp.utils.test_log_like(log_like)

def expsq_kernel(alpha, dx):
    """
    The exponential-squared kernel function. The difference matrix
    can be an arbitrarily shaped numpy array so make sure that you
    use functions like ``numpy.exp`` for exponentiation.
    
    :param alpha: ``(2,)`` The parameter vector ``(amp, ell)``.
    :param dx: ``numpy.array`` The difference matrix. This can be
        a numpy array with arbitrary shape.
    
    :returns K: The kernel matrix (should be the same shape as the
        input ``dx``). 
    
    """
    # Erase the following line and implement your kernel function
    # there.
    pass

gp.utils.test_kernel(expsq_kernel)

gp.interactive.setup_likelihood_sampler(expsq_kernel)

widget = gp.interactive.setup_conditional_sampler(t, y, yerr, expsq_kernel)
widget

def gp_log_prob(params, log_like_fn, kernel_fn, mean_model, t, y, yerr):
    if np.any(params < -10.) or np.any(params > 10.):
        return -np.inf
    
    k = mean_model.vector_size
    
    # Compute the covariance matrix
    K = kernel_fn(np.exp(params[k:]), t[:, None] - t[None, :])
    K[np.diag_indices_from(K)] += yerr**2
    
    # Compute the residual vector
    mean_model.set_parameter_vector(params[:k])
    resid = y - model.get_value(t)
    
    # Compute the log likelihood
    return log_like_fn(resid, K)

ndim = 5
nwalkers = 16
pos = [np.log(np.abs(widget.kwargs[k])) for k in ["depth", "duration", "amp", "ell"]]
pos = np.array(pos[:2] + [widget.kwargs["time"]] + pos[2:]) 
pos = pos + 1e-4*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, gp_log_prob,
                                args=(log_like, expsq_kernel, model, t, y, yerr))
pos, _, _ = sampler.run_mcmc(pos, 200)
sampler.reset()
sampler.run_mcmc(pos, 2000);

gp.corner(sampler.flatchain, truths=np.append(gp.data.true_parameters, [None, None]),
          labels=["ln(depth)", "ln(duration)", "transit time", "ln(amp)", "ln(ell)"])

def matern32_kernel(alpha, dx):
    """
    The Mater-3/2 kernel function. The difference matrix
    can be an arbitrarily shaped numpy array so make sure that you
    use functions like ``numpy.exp`` for exponentiation.
    
    :param alpha: ``(2,)`` The parameter vector ``(amp, ell)``.
    :param dx: ``numpy.array`` The difference matrix. This can be
        a numpy array with arbitrary shape.
    
    :returns K: The kernel matrix (should be the same shape as the
        input ``dx``). 
    
    """
    # Erase the following line and implement your kernel function
    # there.
    pass

widget = gp.interactive.setup_conditional_sampler(t, y, yerr, matern32_kernel)
widget

pos = [np.log(np.abs(widget.kwargs[k])) for k in ["depth", "duration", "amp", "ell"]]
pos = np.array(pos[:2] + [widget.kwargs["time"]] + pos[2:]) 
pos = pos + 1e-4*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, gp_log_prob,
                                args=(log_like, matern32_kernel, model, t, y, yerr))
pos, _, _ = sampler.run_mcmc(pos, 200)
sampler.reset()
sampler.run_mcmc(pos, 2000);

gp.corner(sampler.flatchain, truths=np.append(gp.data.true_parameters, [None, None]),
          labels=["ln(depth)", "ln(duration)", "transit time", "ln(amp)", "ln(ell)"])

t_out, y_out, yerr_out = gp.data.generate_dataset(outliers=75.0)
plt.errorbar(t_out, y_out, yerr=yerr_out, fmt=".k")
plt.xlabel("time [days]")
plt.ylabel("flux [ppm]");

kernel = expsq_kernel
widget = gp.interactive.setup_conditional_sampler(t_out, y_out, yerr_out, kernel)
widget

pos = [np.log(np.abs(widget.kwargs[k])) for k in ["depth", "duration", "amp", "ell"]]
pos = np.array(pos[:2] + [widget.kwargs["time"]] + pos[2:]) 
pos = pos + 1e-4*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, gp_log_prob,
                                args=(log_like, kernel, model, t_out, y_out, yerr_out))
pos, _, _ = sampler.run_mcmc(pos, 200)
sampler.reset()
sampler.run_mcmc(pos, 2000);

gp.corner(sampler.flatchain, truths=np.append(gp.data.true_parameters, [None, None]),
          labels=["ln(depth)", "ln(duration)", "transit time", "ln(amp)", "ln(ell)"])

from scipy.optimize import minimize

# Initial guess
pos = [np.log(np.abs(widget.kwargs[k])) for k in ["depth", "duration", "amp", "ell"]]
pos = np.array(pos[:2] + [widget.kwargs["time"]] + pos[2:])

# Remember that you want to *minimuze* the function
neg_log_prob = lambda *args: -gp_log_prob(*args)

# Implement your sigma clipping procedure here...

# This should be provided to the "args" argument in the minimize function
# In each loop of clipping, update "not_clipped"
not_clipped = np.ones(len(t_out), dtype=bool)
args = (log_like, kernel, model, t_out[not_clipped], y_out[not_clipped], yerr_out[not_clipped])

params = sampler.flatchain[-1]
t_bench, y_bench, yerr_bench = gp.data.generate_dataset(N=2**10)
Ns = 2 ** np.arange(5, 11)
times = np.empty(len(Ns))
for i, N in enumerate(Ns):
    result = get_ipython().magic('timeit -qo gp_log_prob(params, log_like, expsq_kernel, model, t_bench[:N], y_bench[:N], yerr_bench[:N])')
    times[i] = result.best

plt.plot(Ns, times, ".-k", ms=9, label="GP")
plt.plot(Ns, 0.5 * times[-1] * (Ns / Ns[-1])**3, label="N cubed")
plt.xscale("log")
plt.yscale("log")
plt.xlim(Ns.min(), Ns.max())
plt.ylim(0.8 * times.min(), 1.2 * times.max())
plt.legend();



