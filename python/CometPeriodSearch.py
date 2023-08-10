get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style("whitegrid")
sns.set_palette('colorblind')

cdata = "black"
cmodel = "red"

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize as op

from stingray import Lightcurve
from stingray import Powerspectrum

from stingray.pulse import z2_n_probability, z2_n_detection_level, z_n

import emcee
import george
from george import kernels
import corner


apo_dct = pd.read_csv("APO_DCT_A2017U1_2017_10_30_date_mjd_mag_r_mag_unc_obs_code.txt", sep=" ", 
                  names=["time", "mag", "mag_err", "ObsID"])

fig, ax = plt.subplots(1, 1, figsize=(6,4))

ax.errorbar(apo_dct.time, apo_dct.mag, yerr=apo_dct.mag_err, color="black", fmt="o", 
            markersize=5, linewidth=1)

data = apo_dct

class GaussLikelihood(object):
    
    def __init__(self, x, y, yerr, model):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
    def evaluate(self, pars, neg=False):

        mean_model = self.model(self.x, *pars)

        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) -
                         (self.y-mean_model)**2/(2.*self.yerr**2))

        if not np.isfinite(loglike):
            loglike = -np.inf

        if neg:
            return -loglike
        else:
            return loglike
        
    def __call__(self, parameters, neg=False):
        return self.evaluate(parameters, neg)
        

def sinusoid(t, logamp, period, bkg):
    """
    A sinusoidal model.
    
    Parameters
    ----------
    t : iterable
        The dependent coordinate
        
    logamp : float
        The logarithm of the sinusoidal amplitude
        
    period : float
        The period of the sinusoid
        
    phase : float [0, 2*pi]
        The phase of the sinusoidal signal
        
    bkg : float
        The mean magnitude
    
    Returns
    -------
    res : numpy.ndarray
        The result
    """
    res = np.exp(logamp) * np.sin(2.*np.pi*t/period) + bkg
    return res

time = np.array(data["time"])
time -= 0.0
mag = np.array(data["mag"])
mag_err = np.array(data["mag_err"])

period = 4.0/24
amp = 1
bkg = 23.0
phase = 0.5

test_m = sinusoid(time, amp, period, bkg)
test_m2 = sinusoid(time, amp, period+0.01, bkg)


plt.figure(figsize=(6,4))
plt.plot(time, test_m)
plt.plot(time, test_m2)

loglike = GaussLikelihood(time, mag, mag_err, sinusoid)

test_pars = [np.log(amp), period, bkg]

loglike(test_pars, neg=False)

test_pars = [np.log(amp), period+0.01, bkg]

loglike(test_pars, neg=False)

test_pars = [np.log(amp), period,bkg+0.5]
loglike(test_pars, neg=False)

test_pars = [np.log(amp), period,  bkg-0.5]
loglike(test_pars, neg=False)

test_pars = [np.log(amp), period+0.2, bkg]
loglike(test_pars, neg=False)

test_pars = [np.log(amp), period-0.5, bkg]
loglike(test_pars, neg=False)

test_pars = [np.log(amp)-15.0, period-0.5, bkg]
loglike(test_pars, neg=False)

vm = scipy.stats.vonmises(kappa=0.2)

class GaussPosterior(object):
    
    def __init__(self, x, y, yerr, model):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
        self.loglikelihood = GaussLikelihood(x, y, yerr, model)
        #self.vm = scipy.stats.vonmises(kappa=0.2, loc=0.0)
        self.flat_prior = np.log(1/20.0) + np.log(1/(1 - 1/24.0)) +                           np.log(1/5.0) #+ np.log(1.0)
        
    def logprior(self, pars):
        logamp = pars[0]
        period = pars[1]
        #phase = pars[2]
        bkg = pars[2]
        
        if logamp < -20 or logamp > 20:
            return -np.inf
        elif period < 1/24.0 or period > 1.0:
            return -np.inf
        elif bkg < 20 or bkg > 25:
            return -np.inf 
        #elif phase < 0 or phase > 1.0:
        #    return -np.inf
        else:
            return self.flat_prior
        
    def logposterior(self, pars, neg=False):
        lpost = self.logprior(pars) + self.loglikelihood(pars, neg=False)
        
        if not np.isfinite(lpost):
            lpost = -np.inf
            
        if neg:
            return -lpost
        else:
            return lpost
        
    def __call__(self, pars, neg=False):
        return self.logposterior(pars, neg)

lpost = GaussPosterior(time, mag, mag_err, sinusoid)

test_pars = [np.log(amp), period, bkg]

lpost(test_pars)

# should not pass, amplitude out of range:
test_pars = [-21, period, bkg]
print(lpost(test_pars))

# should not pass, amplitude out of range:
test_pars = [30, period, bkg]
print(lpost(test_pars))

##### should not pass, period out of range:
test_pars = [np.log(amp), 0.5/24, bkg]
print(lpost(test_pars))
##### should not pass, period out of range:
test_pars = [np.log(amp), 25/24, bkg]
print(lpost(test_pars))

##### should not pass, phase out of range:
#test_pars = [np.log(amp), period, -0.1, bkg]
#print(lpost(test_pars))
##### should not pass, phase out of range:
#test_pars = [np.log(amp), period, 1.1, bkg]
#print(lpost(test_pars))

##### should not pass, bkg out of range:
test_pars = [np.log(amp), period, 19]
print(lpost(test_pars))
##### should not pass, bkg out of range:
test_pars = [np.log(amp), period, 25.2]
print(lpost(test_pars))

logamp = 0.0
period = 4/24.0
phase = 0.5
bkg = 23.0

fake_pars = [logamp, period, bkg]

mag_model = sinusoid(time, logamp, period, bkg)
fake_mag = mag_model + np.random.normal(loc=0.0, scale=mag_err/2)

plt.figure(figsize=(6,4))

plt.errorbar(time, fake_mag, yerr=mag_err, fmt="o", markersize=5, color="purple")
plt.plot(time, mag_model, color="blue")

fake_lpost = GaussPosterior(time, fake_mag, mag_err, sinusoid)

true_pars = [logamp, period, bkg]

fake_lpost(true_pars, neg=False)

test_pars = [logamp, period+0.4, bkg]
fake_lpost(test_pars, neg=False)

start_pars = [-0.1, 4/24.0, 23.5]

res = op.minimize(fake_lpost, start_pars, args=(True), method="powell")

res.x

model_time = np.linspace(time[0], time[-1], 2000)
m = sinusoid(model_time, *res.x)
input_m = sinusoid(model_time, logamp, period, bkg)

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.errorbar(time, fake_mag, yerr=mag_err, fmt="o", markersize=5, color="purple", label="data")
ax.errorbar(time, mag, yerr=mag_err, fmt="o", markersize=5, color="black", label="data")

ax.plot(model_time, m, color="red", lw=2, label="best-fit model")
ax.plot(model_time, input_m, color="blue", lw=2, label="original model")

# Set up the sampler.
nwalkers, ndim = 200, len(res.x)
sampler = emcee.EnsembleSampler(nwalkers, ndim, fake_lpost, threads=4)

# Initialize the walkers.
p0 = res.x + 0.001 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(fake_lpost(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 5000)

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

names = ["log-amplitude", "period in days", "bkg"]

flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)

corner.corner(flatchain, labels=names, truths=true_pars)

plt.figure(figsize=(8,4))

plt.errorbar(time, fake_mag, yerr=mag_err,
             color="purple", fmt="o", markersize=4)


for i in range(100):
    # Choose a random walker and step.
    w = np.random.randint(sampler.chain.shape[0])
    n = np.random.randint(sampler.chain.shape[1]-100)+100
    p = sampler.chain[w, n]
    m = sinusoid(model_time, *p)
    # Plot a single sample.
    plt.plot(model_time, m, alpha=0.3, color="red")

#plt.xlim(58055.2, 58057)
plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");
plt.gca().invert_yaxis()

start_pars = [0.0, 4/24., 23.3]

res = op.minimize(lpost, start_pars, args=(True), method="powell")

res.x

model_time = np.linspace(time[0], time[-1], 2000)

m = sinusoid(model_time, *res.x)

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.errorbar(time, mag, yerr=mag_err, fmt="o", markersize=5, color="black", label="data")
ax.plot(model_time, m, color="red", lw=2, label="best-fit model")

# Set up the sampler.
nwalkers, ndim = 200, 3
sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, threads=4)

# Initialize the walkers.
p0 = res.x + 0.001 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(lpost(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 10000)

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)

names = ["amplitude", "period in days", "phase", "bkg"]

flatchain[:,0] = np.exp(flatchain[:,0])

fig = plt.figure()
corner.corner(flatchain, labels=names, fig=fig);

np.mean(flatchain[:,0], axis=0)

np.percentile(flatchain[:,0], [50-68.27/2.0, 50, 50+68.27/2.], axis=0)

0.63828670337044935 - 0.59034186

0.68694518 - 0.63828670337044935

flatchain[:, 1] = np.log(flatchain[:,1])

np.mean(flatchain[:,1], axis=0)*24.

np.percentile(flatchain[:,1], [50-(68.27/2.0), 50, 50+(68.27/2.)], axis=0)*24.

4.0659269767275319 - 4.05313703

4.07848433 - 4.0659269767275319

np.mean(flatchain[:,-1], axis=0)

np.percentile(flatchain[:,-1], [50-68.27/2.0, 50, 50+68.27/2.], axis=0)

50+68.27/2.0

phase = time/0.16946752939309517 % 1
phase *= (2.*np.pi)

max_ind = time.searchsorted(58055.5)

sns.set_style("white")

labels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
ticks = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]

0.16946752939309517*24

#fig, (ax1, ax2) = plt.subplots(,2, figsize=(8,4))

fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)


ax1.errorbar(time[:max_ind], mag[:max_ind], yerr=mag_err[:max_ind],
             color="orange", fmt="o", markersize=4, label="APO data")

ax1.errorbar(time[max_ind:], mag[max_ind:], yerr=mag_err[max_ind:],
             color=sns.color_palette("colorblind")[0], fmt="o", markersize=4, 
             lw=1, label="DCT data")


ax2.errorbar(time, mag, yerr=mag_err,
             color=sns.color_palette("colorblind")[0], fmt="o", lw=1, markersize=4, 
             label="DCT data")

flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)
for i in range(30):
    # Choose a random walker and step.
    w = np.random.randint(flatchain.shape[0])
    p = flatchain[w]
    #ph = (p[2]/(2.0*np.pi) - int(p[2]/(2.0*np.pi))) * 2 * np.pi
    m = sinusoid(model_time, p[0], p[1], p[2])
#    # Plot a single sample.
    if i == 0:
        ax1.plot(model_time, m, alpha=0.2, color="black", 
                 label="posterior draws", zorder=0)
        ax2.plot(model_time, m, alpha=0.2, color="black", 
                 label="posterior draws", zorder=0)
    else:
        ax1.plot(model_time, m, alpha=0.2, color="black", zorder=0)
        ax2.plot(model_time, m, alpha=0.2, color="black", zorder=0)


#ax1.set_xlim(58055.2, 58056.32)
ax1.set_xlabel("Time [MJD]")
ax1.set_ylabel("Magnitude");
leg = ax1.legend(frameon=True)
leg.get_frame().set_edgecolor('grey')

ax1.set_ylim(22, 25.5)
ax1.set_ylim(ax1.get_ylim()[::-1])

ax2.set_xlim(58056.17, max(time)+0.01)
ax2.set_xlabel("Time [MJD]")
ax2.set_yticklabels([])
ax2.set_ylim(22, 25.5)
ax2.set_ylim(ax2.get_ylim()[::-1])

ax3.errorbar(phase[:max_ind], mag[:max_ind], yerr=mag_err[:max_ind],
             color="orange", fmt="o", markersize=4, label="APO data")

ax3.errorbar(phase[max_ind:], mag[max_ind:], yerr=mag_err[max_ind:],
             color=sns.color_palette("colorblind")[0], fmt="o", markersize=4, 
             lw=1, label="DCT data")


ax3.set_xticks(ticks)
ax3.set_xticklabels(labels)
ax3.set_title(r"Folded light curve, $P = 4.07$ hours")
ax3.set_ylim(ax3.get_ylim()[::-1])
plt.tight_layout()

plt.savefig("comet_sine_lcs.eps", format="eps")

from period_search import model_sine_curve

lpost, res, sampler = model_sine_curve(data, start_pars, nwalkers=200, niter=5000, nsim=500, 
                                       namestr="test", fitmethod="powell")

sns.color_palette()

plt.figure(figsize=(6,4))
plt.scatter(time, mag, color="black")
plt.scatter(time, fake_mag, color="purple")

kernel = 10 * kernels.ExpSine2Kernel(gamma=1, log_period=np.log(8/24.0),)

gp = george.GP(kernel, mean=np.mean(fake_mag), fit_mean=True,
               white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)
gp.compute(time)

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(fake_mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, fake_mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

#plt.xlim(58054, np.max(time)+0.3)
plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(fake_mag, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(fake_mag, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(time)

# Print the initial ln-likelihood.
print(gp.log_likelihood(fake_mag))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(results.x)
print(gp.log_likelihood(fake_mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(fake_mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, fake_mag, yerr=data["mag_err"],
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

def lnprob(p): 
    mean = p[0]
    logamplitude = p[1]
    loggamma = p[2]
    logperiod = p[3]
    
    if mean < -100 or mean > 100:
    #    print("boo! 0")
        return -np.inf
    
    # prior on log-amplitude: flat and uninformative
    if logamplitude < -100 or logamplitude > 100:
        #print("boo! 1")
        return -np.inf
    
    # prior on log-gamma of the periodic signal: constant and uninformative
    elif loggamma < -20 or loggamma > 20:
        #print("boo! 2")
        return -np.inf
        
    # prior on the period: somewhere between 30 minutes and 2 days
    elif logperiod < np.log(1/24) or logperiod > np.log(23/24.0):
        #print("boo! 4")
        return -np.inf
    
    else:
        pnew = np.array([mean, logamplitude, np.exp(loggamma), logperiod])
        #print("yay!")
        # Update the kernel and compute the lnlikelihood.
        gp.set_parameter_vector(pnew)
        return gp.lnlikelihood(fake_mag, quiet=True)

gp.compute(time)

# Set up the sampler.
nwalkers, ndim = 100, len(results.x)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

# Initialize the walkers.

p0 = results.x + 0.01 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(lnprob(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 10000)

sampler.acceptance_fraction

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

flatchain = np.concatenate(sampler.chain[:,-1000:,:], axis=0)

labels = ["log(Amplitude)", r"$\Gamma$", "log(Period)"]
corner.corner(flatchain[:,1:], labels=labels);

plt.figure(figsize=(6,4))
plt.hist(flatchain[:,-1], bins=200);

kernel = 10 * kernels.ExpSine2Kernel(gamma=10, log_period=np.log(8/24.0),)

gp = george.GP(kernel, mean=np.mean(mag), fit_mean=True,
               white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)
gp.compute(time)

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

#plt.xlim(58054, np.max(time)+0.3)
plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(mag, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(mag, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(time)

# Print the initial ln-likelihood.
print(gp.log_likelihood(mag))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(results.x)
print(gp.log_likelihood(mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=data["mag_err"],
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

kernel = 10 * kernels.ExpSine2Kernel(gamma=10, log_period=np.log(8/24.0),)

gp = george.GP(kernel, mean=np.mean(mag), fit_mean=True,
               white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)
gp.compute(time)

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

#plt.xlim(58054, np.max(time)+0.3)
plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");




# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(mag, quiet=True)

def lnprob(p):
    
    mean = p[0]
    logamplitude = p[1]
    loggamma = p[2]
    logperiod = p[3]
    
    if mean < -100 or mean > 100:
        #print("boo! 0")
        return -np.inf
    
    # prior on log-amplitude: flat and uninformative
    elif logamplitude < -100 or logamplitude > 100:
        #print("boo! 1")
        return -np.inf
    
    # prior on log-gamma of the periodic signal: constant and uninformative
    elif loggamma < -20 or loggamma > 20:
        #print("boo! 2")
        return -np.inf
        
    # prior on the period: somewhere between 30 minutes and 2 days
    elif logperiod < np.log(1/24) or logperiod > np.log(23/24.0):
        #print("boo! 4")
        return -np.inf
    
    else:
        pnew = np.array([mean, logamplitude, np.exp(loggamma), logperiod])
        #print("yay!")
        # Update the kernel and compute the lnlikelihood.
        gp.set_parameter_vector(pnew)
        return gp.lnlikelihood(mag, quiet=True)

pnew = np.zeros(4)
pnew[0] = 23
pnew[1] = 2
pnew[2] = np.log(10)
pnew[3] = np.log(4/24)

lnprob(pnew)

gp.compute(time)

# Set up the sampler.
nwalkers, ndim = 100, len(gp)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

# Initialize the walkers.
p0 = pnew + 0.01 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(lnprob(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 5000)

gp.parameter_names

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

flatchain = np.concatenate(sampler.chain[:,-500:,:], axis=0)

corner.corner(flatchain, labels=["mean magnitude", "log(amplitude)", "log(gamma)", "Sinusoid Period"], bins=100);

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.hist(f, bins=200, histtype="stepfilled");

f = flatchain[:,-1]

f = flatchain[:,-1]

w = np.where(sampler.lnprobability == np.max(sampler.lnprobability))

f = flatchain[:,-1]

w = np.where(sampler.lnprobability == np.max(sampler.lnprobability))

max_ind = [w[0][0], w[1][0]]

f = flatchain[:,-1]

w = np.where(sampler.lnprobability == np.max(sampler.lnprobability))

max_ind = [w[0][0], w[1][0]]

max_pars = sampler.chain[max_ind[0], max_ind[1], :]
# Update the kernel and print the final log-likelihood.
max_pars = [max_pars[0], max_pars[1], np.exp(max_pars[2]), max_pars[3]]

gp.set_parameter_vector(max_pars)
print(gp.log_likelihood(mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=data["mag_err"],
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+2*std, mu-2*std, color="g", alpha=0.5)

plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

new_period = np.exp(max_pars[-1])*2

new_pars = [max_pars[0], max_pars[1], np.exp(max_pars[2]), np.log(new_period)]

flatchain[:,-1] = np.exp(flatchain[:, -1])

flatchain[:,-1] = flatchain[:,-1]*24.0

fig = corner.corner(flatchain[:,1:], labels=["$\log(C)$", r"$\log(1/d^2)$", "$P$ [hours]"], bins=60, smooth=1);
axes = fig.get_axes()
plt.savefig("comet_gp_corner.eps", format="eps", frameon=True)

np.mean(f)

np.percentile(f, [50-68.27/2.0, 50, 50+68.27/2.], axis=0)

4.048004667630849 - 4.00921917

4.08785865 - 4.048004667630849

mind = time.searchsorted(58055.5)

gp.set_parameter_vector(new_pars)
print(gp.log_likelihood(mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

fig, ax = plt.subplots(1, 1, figsize=(8,4))
ax.errorbar(time[:mind], mag[:mind], yerr=mag_err[:mind],lw=1,
             color="orange", fmt="o", markersize=4, label="APO data")

ax.errorbar(time[mind:], mag[mind:], yerr=mag_err[mind:],lw=1, label="DCT data",
             color=sns.color_palette()[0], fmt="o", markersize=4)


ax.fill_between(x, mu+std, mu-std, color="grey", alpha=0.5, label="posterior credible intervals")

leg = ax.legend(frameon=True)
leg.get_frame().set_edgecolor('grey')

ax.set_xlabel("Time [MJD]")
ax.set_ylabel("Magnitude");
ax.set_ylim(ax.get_ylim()[::-1])

plt.tight_layout()
plt.savefig("comet_gp_2periods.pdf", format="pdf")

from period_search import model_gp

pnew

data, kernel, gp, res, sampler = model_gp(data, pnew, 
                                          nwalkers=200, niter=5000, nsim=500, 
                                          namestr="test", fitmethod="powell")



add_data = pd.read_csv("HST_APO_DCT_WHT_GEMN_NOT_WIYN_VLT_GEMS_KECK_CFHT_A2017U1_2017_10_30_date_mjd_mag_r_mag_unc_obs_code_V3.txt", sep="\t", 
                  names=["time", "mag", "mag_err", "ObsID"], skiprows=1)

add_data.head()

fig, ax = plt.subplots(1, 1, figsize=(8,4))
ax.errorbar(add_data.time, add_data.mag, yerr=add_data.mag_err, 
            fmt="o", markersize=5, lw=1, color="black")

sine_lpost, sine_res, sine_sampler = model_sine_curve(add_data, start_pars, nwalkers=400, niter=50000, nsim=100, 
                                       namestr="all_data_with_hst", fitmethod="powell")

gp_data, gp_kernel, gp, gp_res, gp_sampler = model_gp(add_data, pnew, 
                                          nwalkers=250, niter=10000, nsim=200, 
                                          namestr="all_data_with_hst", fitmethod="powell")

obs = ["705", "G37", "250"]

run1_data = add_data[(add_data.ObsID == obs[0]) |
         (add_data.ObsID == obs[1]) |
         (add_data.ObsID == obs[2])]

sine_run1_lpost, sine_run1_res, sine_run1_sampler = model_sine_curve(run1_data, start_pars, nwalkers=400, niter=50000, nsim=100, 
                                       namestr="run1", fitmethod="powell")

gp_run1_data, gp_run1_kernel, gp_run1, gp_run1_res, gp_run1_sampler = model_gp(run1_data, pnew, 
                                          nwalkers=250, niter=10000, nsim=200, 
                                          namestr="run1", fitmethod="powell")







add_data_nodct = add_data#.loc[add_data.ObsID != "G37"]

time = np.array(add_data_nodct.time)
mag = np.array(add_data_nodct.mag)
mag_err = np.array(add_data_nodct.mag_err)

lpost = GaussPosterior(time, mag, mag_err, sinusoid)

start_pars = [0.0, 4/24., 23.3]

res = op.minimize(lpost, start_pars, args=(True), method="powell")

model_time = np.linspace(time[0], time[-1], 2000)

m = sinusoid(model_time, *res.x)

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.errorbar(time, mag, yerr=mag_err, fmt="o", markersize=5, color="black", label="data")
ax.plot(model_time, m, color="red", lw=2, label="best-fit model")

# Set up the sampler.
nwalkers, ndim = 200, 3
sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, threads=4)

# Initialize the walkers.
p0 = res.x + 0.001 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(lpost(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 10000)

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)
names = ["amplitude", "period in days", "bkg"]
flatchain[:,0] = np.exp(flatchain[:,0])

corner.corner(flatchain, labels=names);

np.mean(flatchain[:,1])*24.0

phase = time/0.16374010583452112 % 1
phase *= (2.*np.pi)

max_ind = time.searchsorted(58055.5)

labels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
ticks = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]

unique_obs = add_data.ObsID.unique()

unique_obs

fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))


for i, obs in enumerate(np.array(unique_obs)):
    d = add_data.loc[add_data["ObsID"] == obs]
    t = d.time
    m = d.mag
    me = d.mag_err
    
    ax1.errorbar(t,m, yerr=me,
                 color=sns.color_palette("colorblind", n_colors=len(unique_obs))[i], fmt="o", markersize=4, label=obs)

flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)
for i in range(30):
    # Choose a random walker and step.
    w = np.random.randint(flatchain.shape[0])
    p = flatchain[w]
    m = sinusoid(model_time, p[0], p[1], p[2])
#    # Plot a single sample.
    if i == 0:
        ax1.plot(model_time, m, alpha=0.2, color="black", 
                 label="posterior draws", zorder=0)
        ax2.plot(model_time, m, alpha=0.2, color="black", 
                 label="posterior draws", zorder=0)
    else:
        ax1.plot(model_time, m, alpha=0.2, color="black", zorder=0)
        ax2.plot(model_time, m, alpha=0.2, color="black", zorder=0)


#ax1.set_xlim(58055.2, 58056.32)
ax1.set_xlabel("Time [MJD]")
ax1.set_ylabel("Magnitude");
leg = ax1.legend(frameon=True)
leg.get_frame().set_edgecolor('grey')

ax1.set_ylim(22, 25.5)
ax1.set_ylim(ax1.get_ylim()[::-1])

#ax2.set_xlim(58056.17, max(time)+0.01)
#ax2.set_xlabel("Time [MJD]")
#ax2.set_yticklabels([])
#ax2.set_ylim(22, 25.5)
#ax2.set_ylim(ax2.get_ylim()[::-1])

#ax3.errorbar(phase[:max_ind], mag[:max_ind], yerr=mag_err[:max_ind],
#             color="orange", fmt="o", markersize=4, label="APO data")

#ax3.errorbar(phase[max_ind:], mag[max_ind:], yerr=mag_err[max_ind:],
#             color=sns.color_palette("colorblind")[0], fmt="o", markersize=4, 
#             lw=1, label="DCT data")


#ax3.set_xticks(ticks)
#ax3.set_xticklabels(labels)
#ax3.set_title(r"Folded light curve, $P = 4.07$ hours")
#ax3.set_ylim(ax3.get_ylim()[::-1])
plt.tight_layout()



for i, obs in enumerate(np.array(unique_obs)):
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,4))


    d = add_data.loc[add_data["ObsID"] == obs]
    t = np.array(d.time)
    m = np.array(d.mag)
    me = np.array(d.mag_err)
    
    ax1.errorbar(t,m, yerr=me,
                 color=sns.color_palette("colorblind", n_colors=len(unique_obs))[i], fmt="o", markersize=4, label=obs)

    flatchain = np.concatenate(sampler.chain[:,-1000:, :], axis=0)
    for i in range(30):
        # Choose a random walker and step.
        w = np.random.randint(flatchain.shape[0])
        p = flatchain[w]
        m = sinusoid(model_time, p[0], p[1], p[2])
    #    # Plot a single sample.
        if i == 0:
            ax1.plot(model_time, m, alpha=0.2, color="black", 
                     label="posterior draws", zorder=0)
            ax2.plot(model_time, m, alpha=0.2, color="black", 
                     label="posterior draws", zorder=0)
        else:
            ax1.plot(model_time, m, alpha=0.2, color="black", zorder=0)
            ax2.plot(model_time, m, alpha=0.2, color="black", zorder=0)


    #ax1.set_xlim(58055.2, 58056.32)
    ax1.set_xlabel("Time [MJD]")
    ax1.set_ylabel("Magnitude");
    leg = ax1.legend(frameon=True)
    leg.get_frame().set_edgecolor('grey')

    ax1.set_ylim(22, 25.5)
    ax1.set_ylim(ax1.get_ylim()[::-1])

    ax1.set_xlim(t[0], t[-1])
    ax1.set_title(obs)
    plt.tight_layout()


kernel = 10 * kernels.ExpSine2Kernel(gamma=10, log_period=np.log(8/24.0),)

gp = george.GP(kernel, mean=np.mean(mag), fit_mean=True,
               white_noise=np.mean(np.log(mag_err)), fit_white_noise=False)
gp.compute(time)

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

#plt.xlim(58054, np.max(time)+0.3)
plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(mag, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(mag, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(time)

# Print the initial ln-likelihood.
print(gp.log_likelihood(mag))


# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(results.x)
print(gp.log_likelihood(mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");


def lnprob(p):
    
    mean = p[0]
    logamplitude = p[1]
    loggamma = p[2]
    logperiod = p[3]
    
    if mean < -100 or mean > 100:
        #print("boo! 0")
        return -np.inf
    
    # prior on log-amplitude: flat and uninformative
    elif logamplitude < -100 or logamplitude > 100:
        #print("boo! 1")
        return -np.inf
    
    # prior on log-gamma of the periodic signal: constant and uninformative
    elif loggamma < -20 or loggamma > 20:
        #print("boo! 2")
        return -np.inf
        
    # prior on the period: somewhere between 30 minutes and 2 days
    elif logperiod < np.log(1/24) or logperiod > np.log(23/24.0):
        #print("boo! 4")
        return -np.inf
    
    else:
        pnew = np.array([mean, logamplitude, np.exp(loggamma), logperiod])
        #print("yay!")
        # Update the kernel and compute the lnlikelihood.
        gp.set_parameter_vector(pnew)
        return gp.lnlikelihood(mag, quiet=True)

pnew = np.zeros(4)
pnew[0] = 23
pnew[1] = 2
pnew[2] = np.log(10)
pnew[3] = np.log(4/24)

lnprob(pnew)

gp.compute(time)

# Set up the sampler.
nwalkers, ndim = 100, len(gp)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1)

# Initialize the walkers.
p0 = pnew + 0.01 * np.random.randn(nwalkers, ndim)

for p in p0:
    print(lnprob(p))

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 2000)

for i in range(ndim):
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    ax.plot(sampler.chain[:,:,i].T, color=sns.color_palette()[0], alpha=0.3)

flatchain = np.concatenate(sampler.chain[:,-500:,:], axis=0)

flatchain[:,-1] = np.exp(flatchain[:,-1])

flatchain[:,-1] *= 24

fig = corner.corner(flatchain[:,1:], labels=["$\log(C)$", r"$\log(1/d^2)$", "$P$ [hours]"], bins=60, smooth=1);
axes = fig.get_axes()
#plt.savefig("comet_gp_corner.eps", format="eps", frameon=True)

lnprob = np.concatenate(sampler.lnprobability[:,-500:], axis=0)

lnprob.shape

w

max_pars

f = flatchain[:,-1]

w = np.where(lnprob == np.max(lnprob))


max_ind = w[0][0]

max_pars = np.hstack(flatchain[w])
# Update the kernel and print the final log-likelihood.
max_pars = [max_pars[0], max_pars[1], np.exp(max_pars[2]), max_pars[3]]

gp.set_parameter_vector(max_pars)
print(gp.log_likelihood(mag))

x = np.linspace(np.min(time), np.max(time), 5000)
mu, var = gp.predict(mag, x, return_var=True)
std = np.sqrt(var)

plt.figure(figsize=(8,4))
plt.errorbar(time, mag, yerr=mag_err,
             color="black", fmt="o", markersize=4)
plt.fill_between(x, mu+2*std, mu-2*std, color="g", alpha=0.5)

plt.xlabel("Time [MJD]")
plt.ylabel("Magnitude");



