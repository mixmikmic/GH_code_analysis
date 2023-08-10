import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers, matrix
import pandas as pd

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs)

plt.plot(return_vec.T, alpha=.4);
plt.xlabel('time');
plt.ylabel('returns');

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

print rand_weights(n_assets)
print rand_weights(n_assets)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    wr = np.asmatrix(w*returns)
    
    mu = w * p.T
        
    dist = [abs(mu - x) for x in returns.T * w.T]
    risk = sum(dist)/len(dist)
    
    
    # This recursion reduces outliers to keep plots pretty
    if risk > 1:
        return random_portfolio(returns)
    return mu, risk

n_portfolios = 500
means, risks = np.column_stack([
    random_portfolio(return_vec) 
    for _ in xrange(n_portfolios)
])

plt.plot(risks, means, 'o', markersize=5)
plt.xlabel('risk')
plt.ylabel('mean')
plt.title('Mean and risk of returns of randomly generated portfolios');

def optimal_portfolio(returns, maxrisk):
    '''
    Returns the weights, return, and risk for the return-maximizing portfolio
    subject to a cap on the risk.
    '''
    
    returns = np.asarray(returns)
    n = len(returns)
    time = len(returns[0])
    tot = time + n
    
    # Helper arrays will make assembling LHS easier
    y_indic = [[-1 if j == i else 0 for j in xrange(tot)] for i in xrange(n, tot)]
    x_indic = [[-1 if j == i else 0 for j in xrange(tot)] for i in xrange(n)]
    means = np.mean(returns, axis=1)
    ydef = np.asarray([[returns[i][j]-means[i] if i<n else 0 for i in xrange(tot)] for j in xrange(time)])
    
    # Want to maximize expected returns, which depend on weights*unweighted returns and not on dummy y_t's
    goal = matrix((-1*means).tolist() + [0]*time)
    
    # Linear program coefficients
    RHS = matrix([0]*(2*time + tot) + [maxrisk, 1, -1])
    LHS = (matrix((ydef + y_indic).tolist() + (-1*ydef + y_indic).tolist() + x_indic + y_indic +
               [[0 if j < n else float(1)/time for j in xrange(tot)]] +
               [[1 if j < n else 0 for j in xrange(tot)]] + [[-1 if j < n else 0 for j in xrange(tot)]])).T
    
    sol = solvers.lp(goal, LHS, RHS)
    if sol['status'] == 'primal infeasible' or sol['status'] == 'dual infeasible':
        return None, None, None
    
    # Compute return and risk for the portfolio given by the solver
    w = np.asmatrix((np.array(sol['x'][:n]))).T
    wr = np.asmatrix(w*returns)
    mu = w * np.asmatrix(means).T
    dist = [abs(mu - x) for x in returns.T * w.T]
    risk = sum(dist)/len(dist)
    
    return w.T, mu, risk

# Plot for visualization
plt.plot(risks, means, 'o')
plt.ylabel('mean')
plt.xlabel('risk');
  
for i in range(30):
    cap = .525 - .005*i
    _, mean, risk = optimal_portfolio(return_vec, cap)
    if mean:
        plt.plot(risk, mean, 'y-o');
    else:
        print 'No portfolio with risk cap ' + str(cap) + ' was found.'
        break

print weights

assets = ['XLU', 'XLB', 'XLI', 'XLV', 'XLF', 'XLE', 'MDY', 'XLK', 'XLY', 'XLP', 'QQQ', 'SPY']
data = get_pricing(assets, start_date='2010-01-04', end_date='2015-02-12')

data.loc['price', :, :].plot(figsize=(10,7), colors=['r', 'g', 'b', 'y', 'k', 'c', 'm', 'orange',
                                                     'chartreuse', 'maroon', 'slateblue', 'silver'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('price in $');

rreturns = (data.loc['price', :, :]).pct_change()[1:] # let's not reuse variable names from our simulation
rreturns = np.asarray(rreturns).T

# Generates and computes means and risks for random portfolios
n_portfolios = 500
rmeans, rrisks = np.column_stack([
    random_portfolio(rreturns) 
    for _ in xrange(n_portfolios)
])

fig, ax = plt.subplots()

# Compute and plot means and risks for each of the individual assets
for i in range(12):
    mu = np.mean(rreturns[i,:])
    dist = [abs(mu - x) for x in rreturns[i,:]]
    risk = sum(dist)/len(dist)
    
    ax.plot(risk, mu, 'r-o', markersize = 5)
    ax.annotate(assets[i], (risk, mu))
    
# Sketch out the efficient froniter and print the weights used to achieve efficient portfolios
print ' | '.join(assets), '|| Reward | Risk'
for i in range(21):
    weights, mean, risk = optimal_portfolio(rreturns, (i/20.0)*0.0053 + (1-i/20.0)*0.0083)
    plt.plot(risk, mean, 'y-o');
    print ' '.join(map(lambda x: "%.3f" %x, weights)), "%.5f" %mean[0], "%.4f" %risk[0]
    
plt.plot(rrisks, rmeans, 'o', markersize=5)
plt.xlabel('risk')
plt.ylabel('mean')
plt.title('Mean and risk of returns of randomly generated portfolios');

import zipline
from zipline.api import (add_history, 
                         history, 
                         set_slippage, 
                         slippage,
                         set_commission, 
                         commission, 
                         order_target_percent)

from zipline import TradingAlgorithm


def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading). 
    Use this method to set up any bookkeeping variables.
    
    The context object is passed to all the other methods in your algorithm.

    Parameters

    context: An initialized and empty Python dictionary that has been 
             augmented so that properties can be accessed using dot 
             notation as well as the traditional bracket notation.
    
    Returns None
    '''
    # Register history container to keep a window of the last 100 prices.
    add_history(100, '1d', 'price')
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0
    
def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's 
    securities. 

    Parameters

    data: A dictionary keyed by security id containing the current 
          state of the securities in the algo's universe.

    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state 
             variables defined.

    Returns None
    '''
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = history(100, '1d', 'price').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Perform linear portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T, .04)
        # Rebalance portfolio accordingly
        for stock, weight in zip(prices.columns, weights):
            order_target_percent(stock, weight)
    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass
        
# Instantinate algorithm        
algo = TradingAlgorithm(initialize=initialize, 
                        handle_data=handle_data)
# Run algorithm
results = algo.run(data.swapaxes(2, 0, 1))
results.portfolio_value.plot();

