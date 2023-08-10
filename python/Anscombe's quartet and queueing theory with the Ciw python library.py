get_ipython().magic('matplotlib inline')
import pandas as pd  # Data manipulation
import ciw  # The discrete event simulation library we will use to study queues
import matplotlib.pyplot as plt  # Plots
import seaborn as sns  # Powerful plots
from scipy import stats  # Linear regression
import numpy as np  # Quick summary statistics
import tqdm  # A progress bar

anscombe = sns.load_dataset("anscombe")

print(anscombe)

print(anscombe.groupby("dataset").describe())

for data_set in anscombe.dataset.unique():
    df = anscombe.query("dataset == '{}'".format(data_set))
    slope, intercept, r_val, p_val, slope_std_error = stats.linregress(x=df.x, y=df.y)
    sns.lmplot(x="x", y="y", data=df);
    plt.title("Data set {}: y={:.2f}x+{:.2f} (p: {:.2f}, R^2: {:.2f})".format(data_set, slope, intercept, p_val, r_val))
    plt.savefig("anscombe-{}.png".format(data_set), dpi=400);

parameters = {'Arrival_distributions': {'Class 0': [['Exponential', 0.5]]},   
              'Service_distributions': {'Class 0': [['Exponential', 1]]},
              'Transition_matrices': {'Class 0': [[0.0]]},  
              'Number_of_servers': [1]}

def iteration(parameters, maxtime=250, warmup=50):
    """
    Run a single iteration of the simulation and 
    return the times spent waiting for service 
    as well as the service times
    """
    N = ciw.create_network(parameters)  
    Q = ciw.Simulation(N)
    Q.simulate_until_max_time(maxtime)
    records = [r for r in  Q.get_all_records() if r.arrival_date > warmup]
    waits = [r.waiting_time for r in records]
    service_times = [r.service_time for r in records]
    n = len(waits)
    return waits, service_times

def trials(parameters, repetitions=30, maxtime=2000, warmup=250):
    """Repeat out simulation over a number of trials"""
    waits = []
    service_times = []
    for seed in tqdm.trange(repetitions):  # tqdm gives us a nice progress bar
        ciw.seed(seed)
        wait, service_time = iteration(parameters, maxtime=maxtime, warmup=warmup)
        waits.extend(wait)
        service_times.extend(service_time)
    return waits, service_times

waits, service_times = trials(parameters)

np.mean(service_times)

plt.hist(service_times, normed=True, cumulative=True, histtype = 'step', linewidth=1.5, bins=20, range=[0, 20])
plt.xlabel("Service times")
plt.ylabel("Probability")
plt.title("Cdf");

lmbda = parameters['Arrival_distributions']['Class 0'][0][1]
mu = parameters['Service_distributions']['Class 0'][0][1]
rho = lmbda / mu
np.mean(waits), rho / (mu * (1 - rho))

plt.hist(waits, normed=True, cumulative=True, histtype = 'step', linewidth=1.5, range=[0, 100], bins=20)
plt.xlabel("Waiting time")
plt.ylabel("Probability")
plt.title("Cdf");

distributions = [
    ['Uniform', 0, 2],  # A uniform distribution with mean 1
    ['Deterministic', 1],  # A deterministic distribution with mean 1
    ['Triangular', 0, 2, 1],  # A triangular distribution with mean 1
    ['Exponential', 1],  # The Markovian distribution with mean 1
    ['Gamma', 0.91, 1.1], # A Gamma distribution with mean 1
    ['Lognormal', 0, .1], # A lognormal distribution with mean 1
    ['Weibull', 1.1, 3.9],  # A Weibull distribuion with mean 1
    ['Empirical', [0] * 19 + [20]]  # An empirical distribution with mean 1 (95% of the time: 0, 5% of the time: 20)
]

columns = ["distribution", "waits", "service_times"]
df = pd.DataFrame(columns=columns)  # Create a dataframe that will keep all the data

data = {}
for distribution in distributions:
    parameters['Service_distributions']['Class 0'] = [distribution]
    waits, service_times = trials(parameters)
    n = len(waits)
    df = df.append(pd.DataFrame(list(zip([distribution[0]] * n, waits, service_times)), columns=columns))

bydistribution = df.groupby("distribution")  # Grouping the data
for name, df in sorted(bydistribution, key= lambda dist: dist[1].waits.max()):
    print("{}:\n\t Mean service time: {:.02f}\n\t Mean waiting time: {:.02f}\n\t 95% waiting time: {:.02f} \n\t Max waiting time: {:.02f}\n".format(
                                                                              name, 
                                                                              df.service_times.mean(),
                                                                              df.waits.mean(), 
                                                                              df.waits.quantile(0.95), 
                                                                              df.waits.max()))

for name, df in sorted(bydistribution, key= lambda dist: dist[1].waits.max())[:-1]:
    plt.hist(df.waits, normed=True, bins=20, cumulative=True, 
             histtype = 'step', label=name, linewidth=1.5)
plt.title("Cdf (excluding the empirical distribution)")
plt.xlabel("Waiting time")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend(loc=5);

for name, df in sorted(bydistribution, key= lambda dist: dist[1].waits.max()):
    plt.hist(df.waits, normed=True, bins=20, cumulative=True, 
             histtype = 'step', label=name, linewidth=1.5)
plt.title("Cdf")
plt.xlabel("Waiting time")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend(loc=5);

