import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from IPython.core.pylabtools import figsize
get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.dpi'] = 300
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

def make_model():
    x = pm.Exponential('x', 10) # pre-defined Pymc Exponential distribution
    return locals()

M = pm.MCMC(make_model())

# Parameters of the proposal:
sigma = 1.
# Number of steps
n = 10000

M.use_step_method(pm.Metropolis, M.x, proposal_sd= sigma) # Metropolis Hastings 

M.sample(n, burn = 2000, thin = 2)
samples = M.trace('x')[:]

fig, ax = plt.subplots()
ax.plot(range(samples.shape[0]), samples, lw=1)
ax.set_xlabel('$n$ (steps)')
ax.set_ylabel('$X_n$')

fig, ax = plt.subplots()
idx = np.arange(1, samples.shape[0] + 1)
X_ave = np.cumsum(samples) / idx
ax.plot(idx, X_ave, label='Sampling average of $\mathbb{E}[X_n]$')
ax.plot(idx, 0.10 * np.ones(idx.shape[0]), label='True $\mathbb{E}[X_n]$')
plt.legend(loc='best')
ax.set_xlabel('$m$');

fig, ax = plt.subplots()
X2_ave = np.cumsum(samples ** 2) / idx
X_var = X2_ave - X_ave ** 2
ax.plot(idx, X_var, label='Sampling average of $\mathbb{V}[X_n]$')
ax.plot(idx, 0.01 * np.ones(idx.shape[0]), label='True $\mathbb{V}[X_n]$')
plt.legend(loc='best')
ax.set_xlabel('$m$');

fig, ax = plt.subplots()
ax.hist(samples, normed=True, alpha=0.25, bins=50);
xx = np.linspace(0, 1, 100)
ax.plot(xx, 10. * np.exp(-10. * xx))
ax.set_xlabel('$x$')
ax.set_ylabel('$\pi(x)$');

disaster_data = pd.read_csv('coal_mining_disasters.csv')
disaster_data

plt.scatter(disaster_data.year, disaster_data.disasters, marker= '.')
plt.xlabel('Year')
plt.ylabel('Number of disasters')
plt.title('Recorded coal mining disasters in the UK.')

plt.bar(disaster_data.year, disaster_data.disasters, color="#348ABD")
plt.xlabel("Year")
plt.ylabel("Number of disasters")
plt.title("Recorded coal mining disasters in the UK.")

def make_model(disaster_data):
    """
    PyMC model (wrapping all the data and variables into a single function)
    Inputs:
    disaster_data (111 x 2): Pandas dataframe ( Years vs. Number of coal mining related disasters)
    Outputs:
    lamda_1, lamda_2, tau : stochastic parameters associated with the given model
    """
    
    # Load data
    num_disasters = disaster_data.disasters
    year = disaster_data.year
    n_count_data = disaster_data.shape[0]
    
    # Define Prior
    alpha = 1.0 / np.mean(num_disasters)
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
    
    @pm.deterministic
    def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
        out = np.zeros(n_count_data)
        out[:tau] = lambda_1  # lambda before tau is lambda1
        out[tau:] = lambda_2  # lambda after (and including) tau is lambda2
        return out
    
    # Define Likelihood model
    observation = pm.Poisson("obs", lambda_, value=num_disasters, observed=True)

    return locals()
    

mcmc = pm.MCMC(make_model(disaster_data))
mcmc.sample(40000, 10000, 1)

from pymc.Matplot import plot
plot(mcmc)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

## 4. Plotting Posterior Distributions

plt.figure()
plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="best")
plt.title(r"""Posterior distributions of the variable
    $\lambda_1$""")
plt.xlabel("$\lambda_1$ value")

plt.figure()
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="best")
plt.title(r"""Posterior distributions of the variable
    $\lambda_2$""")
plt.xlabel("$\lambda_2$ value")

plt.figure()
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(1851 + tau_samples,bins=disaster_data.shape[0], alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.);

# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
figsize(12.5, 5)
N = tau_samples.shape[0]
n_count_data = disaster_data.shape[0]
expected_disasters_per_year = np.zeros(n_count_data)
for year in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = year < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_disasters_per_year[year] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.plot(1851 + np.arange(n_count_data), expected_disasters_per_year, lw=4, color="#E24A33",
         label="expected number of disasters")
#plt.xlim(0, n_count_data)
plt.xlabel("Year")
plt.ylabel("Expected number of disasters")
plt.title("Expected number of  coal mining related disasters in an year")
#plt.ylim(0, 60)
#plt.figure()
plt.bar(disaster_data.year, disaster_data.disasters, color="#348ABD")
plt.legend(loc="best");

figsize(12.5, 3.5)
np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt("challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
# drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# plot it, as a function of temperature (the first column)
print("Temp (F), O-Ring failure?")
print(challenger_data)

plt.scatter(challenger_data[:, 0], challenger_data[:, 1])
plt.yticks([0, 1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature");

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

x = np.linspace(-4, 4, 100)

plt.plot(x, logistic(x, 1), label=r"$\beta = 1$", ls="--", lw=1)
plt.plot(x, logistic(x, 3), label=r"$\beta = 3$", ls="--", lw=1)
plt.plot(x, logistic(x, -5), label=r"$\beta = -5$", ls="--", lw=1)

plt.plot(x, logistic(x, 1, 1), label=r"$\beta = 1, \alpha = 1$",
         color="#348ABD")
plt.plot(x, logistic(x, 3, -2), label=r"$\beta = 3, \alpha = -2$",
         color="#A60628")
plt.plot(x, logistic(x, -5, 7), label=r"$\beta = -5, \alpha = 7$",
         color="#7A68A6")

plt.title("Logistic functon with bias", fontsize=14)
plt.legend(loc="lower left");

def make_model(challenger_data):
    """
    PyMC model (wrapping all the data and variables into a single function)
    Inputs:
    challenger data (23 x 2): Pandas dataframe ( Outside Temp vs. Damage Incident)
    Outputs:
    alpha, beta : stochastic parameters associated with the given model
    """
    # Load data
    temp = challenger_data[:, 0]
    temp_scaled = (temp - np.mean(temp))/np.std(temp) # to ensure proper mixing of MCMC chains
    D = challenger_data[:, 1]  # defect or not?
    
    # Define Prior
    beta = pm.Normal("beta", 0, 0.001, value = 0) # 0 mean and small 0.001 precision - vague prior
    alpha = pm.Normal("alpha", 0, 0.001, value = 0)
    
    @pm.deterministic
    def p(t=temp_scaled, alpha=alpha, beta=beta):
        return 1.0 / (1. + np.exp(beta * t + alpha))
    
    # Define likelihood
    # connect the probabilities in `p` with our observations through a
    # Bernoulli random variable.
    observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
    
    return locals()

mcmc = pm.MCMC(make_model(challenger_data))
mcmc.sample(120000, 100000, 2)

from pymc.Matplot import plot
plot(mcmc)

alpha_samples = mcmc.trace('alpha')[:, None]  # best to make them 1d
beta_samples = mcmc.trace('beta')[:, None]

figsize(12.5, 6)

# histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", normed=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", normed=True)
plt.legend();

temp = challenger_data[:, 0]
D = challenger_data[:, 1]

t = np.linspace(temp.min() - 5, temp.max() + 5, 50)[:, None]
t_scaled = (t - np.mean(temp))/np.std(temp)
p_t = logistic(t_scaled.T, beta_samples, alpha_samples)

mean_prob_t = p_t.mean(axis=0)

from scipy.stats.mstats import mquantiles

# vectorized bottom and top 2.5% quantiles for "confidence interval"
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of defect")

plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(temp, D, color="k", s=50, alpha=0.5)
plt.xlabel("temp, $t$")

plt.ylabel("probability estimate")
plt.title("Posterior probability estimates given temp. $t$");

actual_disaster_temp_scaled = (31 - np.mean(temp))/np.std(temp)
prob_31 = logistic(actual_disaster_temp_scaled, beta_samples, alpha_samples)
plt.xlim(0.995, 1)
plt.hist(prob_31, bins=1000, normed=True, histtype='stepfilled')
plt.title("Posterior distribution of probability of defect, given $t = 31$")
plt.xlabel("probability of defect occurring in O-ring");

