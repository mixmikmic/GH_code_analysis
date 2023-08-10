get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import datetime

# Data handler from before
from data_handler import data_handler

# Hide the printouts from the solver
cvxopt.solvers.options['show_progress'] = False

# Let's fetch some stock again
tickers = ['WMT', 'XOM', 'IBM', 'GE']
start_date = '2012-12-31'
end_date = '2013-12-31'
raw_data = data_handler.main(tickers, start_date, end_date, freq='weekly')
data = {}
dates = []
first = True
for ticker in raw_data.keys():
    data[ticker] = []
    for row in sorted(raw_data[ticker], key=lambda x: datetime.datetime.strptime(x['Date'], '%Y-%m-%d')):
        if first:
            dates.append(datetime.datetime.strptime(row['Date'], '%Y-%m-%d').date())
        data[ticker].append(float(row['Adj_Close']))
    first = False

# Compute the returns
returns = {}
for ticker, points in data.items():
    returns[ticker] = np.array([points[i]/points[i-1] for i in range(1, len(points))])

keys = sorted(returns.keys())

# Convert the returns into a 2D array
# TxN matrix where N is the number of stocks
# and T is the number of time steps
returns_array = []
for key in keys:
    returns_array.append(returns[key])

# A helper function to compute mu and std. dev of a stock
def compute_mu_sigma(allocations):
    current_returns = np.zeros(len(dates)-1)
    for i in range(len(keys)):
        current_returns += allocations[i]*returns[keys[i]]
    mean = np.mean(current_returns)
    std = np.std(current_returns, ddof=1)
    return mean, std

# Generate a random portfolio assuming no short sales
def generate_random_portfolio():
    N = len(keys)
    # np.random.rand returns a 1xN matrix of values between [0.0, 1.0)
    allocations = np.random.rand(N)
    allocations /= sum(allocations)
    return allocations    

# Number of random portfolios we want to randomly generate
num_portfolios = 5000

# Seeded so we can replicate the results
np.random.seed(1)

means, stds = np.column_stack(
    [compute_mu_sigma(generate_random_portfolio()) for _ in range(num_portfolios)])

fig = plt.figure(figsize=(15,10))

# Plot the efficient frontier
plt.plot(stds, means, 'o')

plt.xlabel("Standard Deviation ($\sigma$)")
plt.ylabel("Expected Return ($\mathrm{E}[x]$)")
plt.title('Efficient Frontier')

plt.show()

def mvo_no_shorts(rets, m):
    K = len(keys)
    rets = np.asmatrix(rets)
    mu = np.mean(rets, axis=1)
    
    P = np.cov(rets)
    q = [0.0 for _ in range(K)]
    
    # Calling np.transpose on -1.0*mu results in a (K,) array, 
    # we want a (1, K) so we add it to a (1, K) array of zeros
    ret_cond = np.zeros((1, K)) + np.transpose(-1.0*mu)
    G = np.concatenate((ret_cond, -np.eye(K)), axis=0)
    
    # cvxopt complains if m is an integer
    h = [float(-m)] + [0.0 for _ in range(K)]
    
    A = np.ones((1, K))
    b = [1.0]

    sol = cvxopt.solvers.qp(
        cvxopt.matrix(2*P),
        cvxopt.matrix(q),
        cvxopt.matrix(G),
        cvxopt.matrix(h),
        cvxopt.matrix(A),
        cvxopt.matrix(b)
    )
    return sol['x']

# Allocations to achieve a certain portfolio value
allocation = mvo_no_shorts(returns_array, 1.006)
print list(allocation)
print compute_mu_sigma(allocation)

fig = plt.figure(figsize=(15,10))


e_r = np.mean(returns_array, axis=1)
steps = np.linspace(min(e_r), max(e_r), num=100)
allocations = [list(mvo_no_shorts(returns_array, s)) for s in steps]
means1, stds1 = np.column_stack([compute_mu_sigma(alloc) for alloc in allocations])


plt.plot(stds, means, 'o')
plt.plot(stds1, means1, 'r-o')

plt.xlabel("Standard Deviation ($\sigma$)")
plt.ylabel("Expected Return ($\mathrm{E}[x]$)")
plt.title('Efficient Frontier')

plt.show()

# Plot a stacked version of the allocations
from matplotlib.patches import Rectangle
y = np.column_stack(allocations)

percent = y /  y.sum(axis=0) * 100

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)

stack_plt = ax.stackplot(steps, percent)
ax.set_title('MVO Stack Plot')
ax.set_ylabel('Allocation Percentage (%)')
ax.set_xlabel('Expected Return ($\mathrm{E}[x]$)')
ax.margins(0, 0) # Set margins to avoid "whitespace"
proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_plt]
ax.legend(proxy_rects, keys, loc='upper left')
plt.show()

def mvo_shorts(rets, m):
    K = len(keys)
    rets = np.asmatrix(rets)
    mu = np.mean(rets, axis=1)
    
    P = np.cov(rets)
    q = [0.0 for _ in range(K)]
    
    # Calling np.transpose on -1.0*mu results in a (K,) array, 
    # we want a (1, K) so we add it to a (1, K) array of zeros
    G = np.zeros((1, K)) + np.transpose(-1.0*mu)
    
    # cvxopt complains if m is an integer
    h = [float(-m)]
    
    A = np.ones((1, K))
    b = [1.0]

    sol = cvxopt.solvers.qp(
        cvxopt.matrix(2*P),
        cvxopt.matrix(q),
        cvxopt.matrix(G),
        cvxopt.matrix(h),
        cvxopt.matrix(A),
        cvxopt.matrix(b)
    )
    return sol['x']

allocation = mvo_shorts(returns_array, 1.008)
print list(allocation)
print compute_mu_sigma(allocation)

allocations = [list(mvo_shorts(returns_array, s)) for s in steps]
means2, stds2 = np.column_stack([compute_mu_sigma(alloc) for alloc in allocations])

fig = plt.figure(figsize=(15,10))
h1, = plt.plot(stds1, means1, 'r-o', label='No Short Sales')
h2, = plt.plot(stds2, means2, 'b-o', label='With Short Sales')
plt.xlabel('Standard Deviation (\sigma)')
plt.ylabel('Expected Return ($\mathrm{E}[x]$)')
plt.legend([h1, h2], loc='upper left', numpoints=1)

def generate_rnd_port():
    N = len(keys)
    # np.random.rand returns a 1xN matrix of values between [0.0, 1.0)
    allocations = 2*np.random.rand(N)-1
    allocations /= sum(allocations)
    return allocations

# Number of random portfolios we want to randomly generate
num_portfolios = 10000

# Seeded so we can replicate the results
np.random.seed(1)

# I am filtering out some of the samples so everything is in the same range
means3, stds3 = np.column_stack(filter(lambda x: x[0] < max(e_r) and x[0] > min(e_r) and x[1] < 0.02, 
    [compute_mu_sigma(generate_rnd_port()) for _ in range(num_portfolios)]))

fig = plt.figure(figsize=(15,10))

# Plot the efficient frontier
h0, = plt.plot(stds3, means3, 'o', label='Random Portfolios w/ shorting')
h1, = plt.plot(stds1, means1, 'r-o', label='No Short Sales')
h2, = plt.plot(stds2, means2, 'g-o', label='With Short Sales')

plt.xlabel("Standard Deviation ($\sigma$)")
plt.ylabel("Expected Return ($\mathrm{E}[x]$)")
plt.title('Efficient Frontier')
plt.legend([h0, h1, h2], loc='upper left', numpoints=1)

plt.show()



