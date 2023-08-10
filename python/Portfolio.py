import pandas as pd
import numpy as np
import quandl
from scipy.optimize import minimize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Define a Start and End Date of Historical Data
start = pd.to_datetime("2012-01-01")
end = pd.to_datetime("2017-01-01")

# Pull data from using Quandl
aapl = quandl.get('WIKI/AAPL.11', start_date = start, end_date = end)
cisco = quandl.get('WIKI/CSCO.11', start_date = start, end_date = end)
ibm = quandl.get('WIKI/IBM.11', start_date = start, end_date = end)
amzn = quandl.get('WIKI/AMZN.11', start_date = start, end_date = end)

aapl.head(10)

aapl.iloc[0]['Adj. Close']

# Cumulative Return against Day 1

for stock_df in (aapl, cisco, ibm, amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close'] / stock_df.iloc[0]['Adj. Close']

aapl.head(10)

aapl.tail(10)

# Assumed Allocation: 30% AAPL, 20% CISCO, 40% AMAZON, 10% IBM
for stock_df, allo in zip((aapl, cisco, ibm, amzn), [.3, .2, .4, .1]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo

aapl.head(10)

# Assumed Deployed Capital: $1,000,000
# Calculate Return in $

for stock_df in (aapl, cisco, ibm, amzn):
    stock_df['Position Values'] = stock_df['Allocation']*1000000

aapl.head(10)

all_pos_vals = [aapl['Position Values'], cisco['Position Values'],
               ibm['Position Values'], amzn['Position Values']]
portfolio_val = pd.concat(all_pos_vals, axis = 1)
portfolio_val.columns = ['AAPL Pos', 'CISCO Pos', 'IBM Pos', 'AMZN Pos']

portfolio_val.head()

portfolio_val['Total Pos'] = portfolio_val.sum(axis = 1)

portfolio_val.head()

portfolio_val['Total Pos'].plot(figsize = (10, 8));
plt.title('Total Portfolio Value ($)');

# Plot all Companies Individually 

portfolio_val.drop('Total Pos', axis = 1).plot(figsize = (10, 8));
plt.title('Individual Portfolio Return Performance ($)');

portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)
portfolio_val.head()

portfolio_val['Daily Return'].plot(kind = 'hist', bins = 100, figsize = (4,5));

portfolio_val['Daily Return'].plot(kind = 'kde', figsize = (4,5));

portfolio_val['Daily Return'].mean()

portfolio_val['Daily Return'].std()

cumulative_return = 100*(portfolio_val['Total Pos'][-1]/portfolio_val['Total Pos'][0]-1)
cumulative_return

SR = portfolio_val['Daily Return'].mean() / portfolio_val['Daily Return'].std()
SR

# Annualized Sharpe Ratio

ASR = SR * (252**0.5)
ASR

# Create Data Frame
stocks = pd.concat([aapl.iloc[: , 0], cisco.iloc[: , 0], ibm.iloc[: , 0], amzn.iloc[: , 0]], axis = 1)
stocks.columns = ['aapl', 'cisco', 'ibm', 'amzn']
stocks.head()

# simulate daily return

stocks.pct_change(1).mean()

# Correlation between stocks
stocks.pct_change(1).corr()

# Transform to Logarithmic Return for Normality (many analysis assume this property)
log_ret = np.log(stocks / stocks.shift(1))
log_ret.head()

log_ret.hist(bins = 100, figsize = (11,8));
plt.tight_layout()

log_ret.mean()

log_ret.cov()

# Sharpe Ratio Randomization
np.random.seed(101)
num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    
    # Weights
    weights = np.array(np.random.random(4))
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind, :] = weights

    # Expected Average Return based on 2-year historical data
    ret_arr[ind] = np.sum((log_ret.mean()*weights)*252)

    # Expected Volatility (Variance)
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

sharpe_arr.max()

sharpe_arr.argmax()

all_weights[1420, :]

# Plot the Efficient Frontier
plt.figure(figsize = (12, 8));
plt.scatter(vol_arr, ret_arr, c = sharpe_arr, cmap = 'plasma');
plt.colorbar(label = 'Sharpe Ratio');
plt.xlabel('Volatility');
plt.ylabel('Return');

# Add Red Dots for the Max Return
plt.scatter(vol_arr[sharpe_arr.argmax()],ret_arr[sharpe_arr.argmax()], c = 'red', s = 50, edgecolors = 'black');

# Variables: weights

def get_ret_vol_sr(weights):
    '''
    This function returns an array of portfolio return, volatility, and Sharpe Ratio. It uses the user specified 
    allocation weights and 2-year historical stock return to calculate Volatility and SR
    Input: Weights of allocation
    Output: an array of return, volatility, and SR
    '''
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

# Objective Function: Sharpe Ratio based on Input Weights

def neg_sharpe(weights):
    '''
    This is a helper function for the Optimizer. We want to maximize Sharpe by minimizing the 
    negative of the Sharpe by multipling by -1
    '''
    return get_ret_vol_sr(weights)[2] * -1

# Constraints

def check_sum(weights):
    '''
    This is a helper function for constraints. Return 0 if the sum of the weights is 1.
    '''
    return np.sum(weights) - 1

cons = ({'type':'eq', 'fun':check_sum})

# Boundary Conditions of Input Variables (weights)

bounds = ((0,1), (0,1), (0,1), (0,1))

# Initial Conditions (even weights)

init_guess = [0.25, 0.25, 0.25, 0.25]

# Run the Optimization

opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds = bounds, constraints=cons)
opt_results

# Find the Optimal Weights that Lead to the Maximum Sharpe Ratio

opt_results.x

# Return the Portfolio Return, Volatility, and Sharpe Ratio

get_ret_vol_sr(opt_results.x)

