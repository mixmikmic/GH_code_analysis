import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import corner
import emcee

def model(pars, x):
    return pars[0] * x + pars[1]

def lnprior(pars):
    if -10 < pars[0] < 10 and -10 < pars[1] < 10 and -10 < pars[2] < 10:
        return 0.
    else:
        return -np.inf

def lnprob(pars, x, y, yerr):
    return lnprior(pars) + lnlike(pars, x, y, yerr)

def lnlike(pars, x, y, yerr):
    invsig2 = 1./(yerr**2 + np.exp(2*pars[2]))
    model_y = model(pars, x)
    return -.5*np.sum((y-model_y)**2*invsig2 - np.log(invsig2))

# load data
f, ferr, r, rerr = np.genfromtxt("../data/flickers.dat").T

# fit a line
AT = np.vstack((r, np.ones_like(r)))
ATA = np.dot(AT, AT.T)
m, c = np.linalg.solve(ATA, np.dot(AT, f))
print("params = ", m, c)

# plot data with best fit line
xs = np.linspace(min(r), max(r), 100)
ys = m * xs + c
plt.errorbar(r, f, xerr=rerr, yerr=ferr, fmt="k.", capsize=0)
plt.plot(xs, ys)
plt.ylabel("log flicker")
plt.xlabel("log rho")

pars_init = [m, c, np.log(3)]
ndim, nwalkers = 3, 24
pos = [pars_init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]  # initialisation
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(r, f, ferr))
sampler.run_mcmc(pos, 10000)  # run MCMC
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))  # cut off burn in and flatten chains
samples[:, 2] = np.exp(samples[:, 2])

m1, c1, sig1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

fig = corner.corner(samples, labels=["m", "c", "s"])
#print(sig1[0])
#print(1./(ferr**2 + np.exp(2*pars[2])))

def analytic_lnlike(pars, x, y, xerr, yerr):
    m, b, lns = pars
    sig2 = (m*xerr)**2 + (yerr**2 + np.exp(2*lns))
    ll = -.5 * np.sum((m*x-y+b)**2/sig2 + np.log(sig2))
    return ll

def analytic_lnprob(pars, x, y, xerr, yerr):
    return lnprior(pars) + analytic_lnlike(pars, x, y, xerr, yerr)

pars_init = [m, c, np.log(3)]
ndim, nwalkers = 3, 24
pos = [pars_init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]  # initialisation
sampler = emcee.EnsembleSampler(nwalkers, ndim, analytic_lnprob, args=(r, f, rerr, ferr))
sampler.run_mcmc(pos, 10000)  # run MCMC
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))  # cut off burn in and flatten chains
samples[:, 2] = np.exp(samples[:, 2])
m2, c2, ln_sig2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

fig = corner.corner(samples, labels=["m", "c", "s"])

print(m1)
print(m2)
print((m1[0] - m2[0])/ m1[1])

print(c1)
print(c2)
print((c1[0] - c2[0])/ c1[1])

print(sig1)
print(ln_sig2)
print((sig1[0] - ln_sig2[0])/ sig1[1])

# load data
f, ferr, l, lerr, _, _ = np.genfromtxt("../data/log.dat").T

# fit a line
AT = np.vstack((l, np.ones_like(r)))
ATA = np.dot(AT, AT.T)
m, c = np.linalg.solve(ATA, np.dot(AT, f))
print("params = ", m, c)

# plot data with best fit line
xs = np.linspace(min(l), max(l), 100)
ys = m * xs + c
plt.errorbar(l, f, xerr=lerr, yerr=ferr, fmt="k.", capsize=0)
plt.plot(xs, ys)
plt.ylabel("log flicker")
plt.xlabel("log g")

pars_init = [m, c, np.log(3)]
ndim, nwalkers = 3, 24
pos = [pars_init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]  # initialisation
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(l, f, ferr))
sampler.run_mcmc(pos, 10000)  # run MCMC
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))  # cut off burn in and flatten chains
samples[:, 2] = np.exp(samples[:, 2])

m1, c1, ln_sig1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

fig = corner.corner(samples, labels=["m", "c", "s"])

pars_init = [m, c, np.log(3)]
ndim, nwalkers = 3, 24
pos = [pars_init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]  # initialisation
sampler = emcee.EnsembleSampler(nwalkers, ndim, analytic_lnprob, args=(l, f, rerr, ferr))
sampler.run_mcmc(pos, 10000)  # run MCMC
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))  # cut off burn in and flatten chains
samples[:, 2] = np.exp(samples[:, 2])
m2, c2, ln_sig2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

fig = corner.corner(samples, labels=["m", "c", "s"])

print(m1)
print(m2)
print((m1[0] - m2[0])/ m1[1])

print(c1)
print(c2)
print((c1[0] - c2[0])/ c1[1])

print(sig1)
print(ln_sig2)
print((sig1[0] - ln_sig2[0])/ sig1[1])



