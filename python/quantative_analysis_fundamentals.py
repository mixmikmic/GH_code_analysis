import pandas as pd
performance = pd.read_pickle('results/fundamentals_long.pickle')
performance_excluded = pd.read_pickle('results/excluded_fundamentals.pickle')
performance_alc = pd.read_pickle('results/alcohol_fundamentals.pickle')
performance_alcCoal = pd.read_pickle('results/alcCoal_fundamentals.pickle')
performance_alcTabCoal = pd.read_pickle('results/alcTabCoal_fundamentals.pickle')
performance_alcTab = pd.read_pickle('results/alcTab_fundamentals.pickle')
performance_coal = pd.read_pickle('results/coal_fundamentals.pickle')
performance_defAlcCoal = pd.read_pickle('results/defAlcCoal_fundamentals.pickle')
performance_defAlc = pd.read_pickle('results/defAlc_fundamentals.pickle')
performance_defAlcTab = pd.read_pickle('results/defAlcTab_fundamentals.pickle')
performance_defCoal = pd.read_pickle('results/defCoal_fundamentals.pickle')
performance_def = pd.read_pickle('results/defense_fundamentals.pickle')
performance_defTabCoal = pd.read_pickle('results/defTabCoal_fundamentals.pickle')
performance_defTab = pd.read_pickle('results/defTab_fundamentals.pickle')
performance_tab = pd.read_pickle('results/tabacco_fundamentals.pickle')
performance_tabCoal = pd.read_pickle('results/tabCoal_fundamentals.pickle')

# display the rows that we have in the dataset
for row in performance.columns.values:
    print(row)

get_ipython().magic('pylab inline')
figsize(12, 12)
import matplotlib.pyplot as plt

fig_return_1 = plt.figure()
return_graph_1 = fig_return_1.add_subplot(211)
bench_performance = plt.plot(performance.benchmark_period_return, label='Benchmark return')
algo_performance = plt.plot(performance.algorithm_period_return, label='Unrestricted return')
algo_performance_alc = plt.plot(performance_alc.algorithm_period_return, label='Blacklisted alcohol return')
algo_performance_coal = plt.plot(performance_coal.algorithm_period_return, label='Blacklisted coal return')
algo_performance_def = plt.plot(performance_def.algorithm_period_return, label='Blacklisted defence return')
algo_performance_tab = plt.plot(performance_tab.algorithm_period_return, label='Blacklisted tabacco return')
plt.legend(loc=0)
plt.show()

fig_return_2 = plt.figure()
return_graph_2 = fig_return_2.add_subplot(211)
bench_performance = plt.plot(performance.benchmark_period_return, label='Benchmark return')
algo_performance_alcCoal = plt.plot(performance_alcCoal.algorithm_period_return, label='Blacklisted alcohol and coal return')
algo_performance_alcTab = plt.plot(performance_alcTab.algorithm_period_return, label='Blacklisted alcohol and tabacco return')
algo_performance_defAlc = plt.plot(performance_defAlc.algorithm_period_return, label='Blacklisted defence and alcohol return')
algo_performance_defCoal = plt.plot(performance_defCoal.algorithm_period_return, label='Blacklisted defence and coal return')
algo_performance_defTab = plt.plot(performance_defTab.algorithm_period_return, label='Blacklisted defence and tabacco return')
algo_performance_tabCoal = plt.plot(performance_tabCoal.algorithm_period_return, label='Blacklisted tabacco and coal return')
plt.legend(loc=0)
plt.show()

fig_return_3 = plt.figure()
return_graph_3 = fig_return_3.add_subplot(211)
bench_performance = plt.plot(performance.benchmark_period_return, label='Benchmark return')
algo_performance_alcTabCoal = plt.plot(performance_alcTabCoal.algorithm_period_return, label='Blacklisted alcohol, tabacco, and coal return')
algo_performance_defAlcCoal = plt.plot(performance_defAlcCoal.algorithm_period_return, label='Blacklisted defence, alcohol, and coal return')
algo_performance_defAlcTab = plt.plot(performance_defAlcTab.algorithm_period_return, label='Blacklisted defence, alcohol, and tabacco return')
algo_performance_defTabCoal = plt.plot(performance_defTabCoal.algorithm_period_return, label='Blacklisted defence, tabacco, and coal return')
algo_performance_excluded = plt.plot(performance_excluded.algorithm_period_return, label='Blacklisted defence, alcohol, tabacco, and coal return')
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
ax1.plot(performance_excluded.sharpe, 'b', label="Sharpe")
plt.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(performance_excluded.algo_volatility, 'g', label="Algorithm volatility")
ax2.plot(performance_excluded.algorithm_period_return, 'r', label="Algorithm return")
ax2.plot(performance_excluded.benchmark_period_return, 'y', label="Benchmark return")
plt.legend(loc=1)
plt.show()

alpha_graph, ax1 = plt.subplots()
ax1.plot(performance_excluded.alpha, 'b')
plt.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(performance_excluded.beta, 'r')
plt.legend(loc=1)
plt.show

monthly_returns = performance_excluded.returns.resample('M', how='mean')
weekly_returns = performance_excluded.returns.resample('W', how='mean').dropna()
# replace NaN values for plotting with default return (0)
daily_returns = performance_excluded.returns.fillna(value=0)

monthly_sharpe = performance_excluded.sharpe.resample('M', how='mean')
weekly_sharpe = performance_excluded.sharpe.resample('W', how='mean').dropna()
# drop NaN values for plotting
daily_sharpe = performance_excluded.sharpe.dropna()

monthly_alpha = performance_excluded.alpha.resample('M', how='mean')
weekly_alpha = performance_excluded.alpha.resample('W', how='mean').dropna()
# drop NaN values for plotting
daily_alpha = performance_excluded.alpha.dropna()

monthly_beta = performance_excluded.beta.resample('M', how='mean')
weekly_beta = performance_excluded.beta.resample('W', how='mean').dropna()
# drop NaN values for plotting
daily_beta = performance_excluded.beta.dropna()

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

