import pandas as pd
performance = pd.read_pickle('results/momentum_pipeline.pickle')
# display the top 5 rows to see if the load worked
performance.head()

# display the rows that we have in the dataset
for row in performance.columns.values:
    print(row)

get_ipython().magic('pylab inline')
figsize(12, 12)
import matplotlib.pyplot as plt

fig = plt.figure()
return_graph1 = fig.add_subplot(211)
algo_performance = plt.plot(performance.algorithm_period_return)
bench_performance = plt.plot(performance.benchmark_period_return)
plt.legend(loc=0)
plt.show()

return_graph2 = fig.add_subplot(212)
algo_return = plt.plot(performance.ending_cash)
algo_long = plt.plot(performance.long_value)
# Take inverse of short value for comparison (by default short is always negative)
algo_short = plt.plot(-performance.short_value)
plt.legend(loc=0)
plt.show()

return_graph3, ax1 = plt.subplots()
ax1.plot(performance.sharpe, 'b')
plt.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(performance.algo_volatility, 'g')
ax2.plot(performance.algorithm_period_return, 'r')
ax2.plot(performance.benchmark_period_return, 'y')
plt.legend(loc=1)
plt.show()

alpha_graph, ax1 = plt.subplots()
ax1.plot(performance.alpha, 'b')
plt.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(performance.beta, 'r')
plt.legend(loc=1)
plt.show

monthly_returns = performance.returns.resample('M', how='mean')
weekly_returns = performance.returns.resample('W', how='mean')
# replace NaN values for plotting with default return (0)
daily_returns = performance.returns.fillna(value=0)
print('Monthly')
print(monthly_returns)
print('\nWeekly')
print(weekly_returns)
print('\nDaily')
print(daily_returns)

monthly_sharpe = performance.sharpe.resample('M', how='mean')
weekly_sharpe = performance.sharpe.resample('W', how='mean')
# drop NaN values for plotting
daily_sharpe = performance.sharpe.dropna()
print('Monthly')
print(monthly_sharpe)
print('\nWeekly')
print(weekly_sharpe)
print('\nDaily')
print(daily_sharpe)

monthly_alpha = performance.alpha.resample('M', how='mean')
weekly_alpha = performance.alpha.resample('W', how='mean')
# drop NaN values for plotting
daily_alpha = performance.alpha.dropna()
print('Monthly')
print(monthly_alpha)
print('\nWeekly')
print(weekly_alpha)
print('\nDaily')
print(daily_alpha)

monthly_beta = performance.beta.resample('M', how='mean')
weekly_beta = performance.beta.resample('W', how='mean')
# drop NaN values for plotting
daily_beta = performance.beta.dropna()
print('Monthly')
print(monthly_beta)
print('\nWeekly')
print(weekly_beta)
print('\nDaily')
print(daily_beta)

fig, axes = plt.subplots(nrows=2, ncols=2)
labels = ['monthly', 'weekly', 'daily']
axes[0, 0].boxplot((monthly_returns, weekly_returns, daily_returns), labels=labels, showmeans=True)
axes[0, 0].set_title('Return')
axes[0, 1].boxplot((monthly_sharpe, weekly_sharpe, daily_sharpe), labels=labels, showmeans=True)
axes[0, 1].set_title('Sharpe')
axes[1, 0].boxplot((monthly_alpha, weekly_alpha, daily_alpha), labels=labels, showmeans=True)
axes[1, 0].set_title('Alpha')
axes[1, 1].boxplot((monthly_beta, weekly_beta, daily_beta), labels=labels, showmeans=True)
axes[1, 1].set_title('Beta')
plt.setp(axes)
plt.show()

from statistics import stdev
print('Standard deviation\nWeekly Sharpe: {}\nDaily: Sharpe {}'.format(stdev(weekly_sharpe), stdev(daily_sharpe)))

