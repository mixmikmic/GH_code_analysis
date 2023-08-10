import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('VIX Study.csv', parse_dates=True, index_col=0).sort_index(ascending=True).round(2)
df['SPY returns'] = df['SPY Close'].pct_change()
df['VIX returns'] = df['VIX'].pct_change()
df['fwd returns'] = df['SPY Open'].pct_change().shift(-2)
df['hist vol'] = df['SPY returns'].rolling(21).std() * np.sqrt(252) * 100
df['fwd vol'] = df['SPY returns'].rolling(21).std().shift(-21) * np.sqrt(252) * 100
df.head(8)

volatilities = df[['VIX', 'hist vol', 'fwd vol']].dropna()
volatilities.corr()

sns.pairplot(volatilities)

df[['SPY returns', 'VIX returns']].corr()

vix_fwd = volatilities['VIX'] - volatilities['fwd vol']
vix_fwd.hist(bins=20)
plt.title('Implied Volatility - Realized Volatility')
plt.xlabel('% Difference')
plt.ylabel('Occurences')

vix_fwd.describe()

bottom_percentile = np.percentile(vix_fwd, 2.5)
worst_days = vix_fwd < bottom_percentile
df.dropna()['SPY Close'].plot()
df.dropna().loc[worst_days, 'SPY Close'].plot(style='ro')

vix_hist = volatilities['VIX'] - volatilities['hist vol']
vix_hist.hist(bins=20)

vix_hist_low = np.percentile(vix_hist, 1)
vix_hist_high = np.percentile(vix_hist, 99)
vix_hist_low_days = vix_hist <= vix_hist_low
vix_hist_high_days = vix_hist > vix_hist_high
df.dropna()['SPY Close'].plot()
df.dropna().loc[vix_hist_low_days, 'SPY Close'].plot(style='ro')
df.dropna().loc[vix_hist_high_days, 'SPY Close'].plot(style='go')

df['proj upper'] = df['SPY Close'].shift(21) * (1 + df['VIX'].shift(21) / 100 * np.sqrt(21) / np.sqrt(252))
df['proj lower'] = df['SPY Close'].shift(21) * (1 - df['VIX'].shift(21) / 100 * np.sqrt(21) / np.sqrt(252))
df.loc['2008', ['SPY Close', 'proj upper', 'proj lower']].plot(style=['b-', 'g:', 'r:'])

df.loc['2017', ['SPY Close', 'proj upper', 'proj lower']].plot(style=['b-', 'g:', 'r:'])

n = 5
expected_num = df['SPY Close'] - df['SPY Close'].shift(n)
expected_denom = df['SPY Close'].shift(n) * df['VIX'].shift(n) / 100 * np.sqrt(n) / np.sqrt(252)
df['vs expected'] = expected_num / expected_denom
df['vs expected'].hist()

df.loc[df['vs expected']>0, 'vs expected'].mean()

df.loc[df['vs expected']<=0, 'vs expected'].mean()

df.loc[df['vs expected']>0, 'vs expected'].count() / df['vs expected'].count()

import scipy.stats as stats
import pylab
stats.probplot(df['vs expected'].dropna(), dist='norm', plot=pylab)

df.loc['2017', 'vs expected'].plot()

df['fwd returns'].groupby(pd.qcut(df['vs expected'], 10)).mean().plot(kind='bar')

eq = (1 + df.loc[df['vs expected'] < -0.5, 'fwd returns']).cumprod()
eq.plot()
(1 + df['fwd returns']).cumprod().plot()

def sharpe_ratio(returns):
    return returns.mean() / returns.std()

def drawdown(eq):
    return (eq / eq.cummax() - 1)

print(sharpe_ratio(df['fwd returns']) * np.sqrt(252))
print(sharpe_ratio(df.loc[df['vs expected'] < -0.5, 'fwd returns']) * np.sqrt(252))

drawdown(eq).plot()
df['SPY DD'] = drawdown(df['SPY Close'])
df['SPY DD'].plot()

strat_ulcer = np.sqrt(((eq / eq.cummax() - 1)**2).mean()) * 100
spy_ulcer = np.sqrt(((df['SPY Close'] / df['SPY Close'].cummax() - 1)**2).mean()) * 100
print(strat_ulcer)
print(spy_ulcer)

over_40 = df.loc[df['VIX']>40]

df['2008']['SPY Close'].plot()
df.loc[over_40.index, 'SPY Close']['2008'].plot(style='ro')
df['SPY Close'].rolling(200).mean()['2008'].plot()

over_40.loc[over_40['fwd returns'] <= 0, 'fwd returns'].describe()

over_vix_under_ma = (df['VIX'] > 35) & (df['SPY Close'] > df['SPY Close'].rolling(200).mean())
df.loc[over_vix_under_ma, 'fwd returns']

def summary_stats(returns):
    stats = pd.Series()
    gains = returns[returns > 0]
    losses = returns[returns <= 0]
    num_total = len(returns)
    num_gains = len(gains)
    num_losses = len(losses)
    avg = np.mean(returns)
    volatility = np.std(returns)
    sharpe = avg / volatility
    win_pct = num_gains / num_total
    avg_win = np.mean(gains)
    avg_loss = np.mean(losses)
    stats['total returns'] = num_total
    stats['total gains'] = num_gains
    stats['total losses'] = num_losses
    stats['expectancy (%)'] = avg * 100
    stats['volatilty (%)'] = volatility * 100
    stats['sharpe (daily)'] = sharpe
    stats['win %'] = win_pct * 100
    stats['total returns'] = num_total    
    stats['average gain (%)'] = avg_win * 100    
    stats['average loss (%)'] = avg_loss * 100
    return stats
    

strat = df.loc[df['VIX'] < df['hist vol'], 'fwd returns']
pd.DataFrame({'strat':summary_stats(strat), 'SPY':summary_stats(df['fwd returns'])})

import statsmodels.api as sm
X = df.dropna()['hist vol']
X = sm.add_constant(X)
y = df.dropna()['VIX']
model = sm.OLS(y, X).fit()
model.params

model.summary()

historical_component = df['hist vol'] * model.params[1] + model.params[0]
plt.scatter(df['hist vol'], df['VIX'])
plt.plot(df['hist vol'], historical_component, color='r')

# plot residuals
resid = df['VIX'] - historical_component
resid.plot()
#plt.axhline(resid.mean(), color='r')
plt.title('VIX Residuals')

def obj_func(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def bound(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

normalized = normalize(resid)
bounded = bound(normalized)
bounded = bounded * 2 - 1
bounded.plot()
plt.title('Scaled Residuals')

print(len(bounded))
print(len(df['fwd returns']))

df['fwd returns'].groupby(pd.qcut(bounded, 10)).mean().plot(kind='bar')
plt.title('Mean Return by Excess VIX Decile')
plt.xlabel('Scaled Excess VIX')
plt.ylabel('Mean Daily SPY Return')

compare_df = pd.DataFrame()
compare_df['SPY'] = summary_stats(df['fwd returns'])
compare_df['Top Decile'] = summary_stats(df.loc[bounded > bounded.quantile(0.9), 'fwd returns'])
compare_df

expanding_quantile = bounded.expanding(min_periods=10).quantile(0.9)
top_quantile = bounded > expanding_quantile
filtered = df.loc[top_quantile, 'fwd returns']
filtered_2 = df.loc[df['VIX']>30, 'fwd returns']
filtered_3 = df.loc[bounded > 0, 'fwd returns']
results = pd.DataFrame()
results['SPY'] = summary_stats(df['fwd returns'])
results['Top Quantile'] = summary_stats(filtered)
results['Excess > 0'] = summary_stats(filtered_3)
results['High VIX'] = summary_stats(filtered_2)
(1 + filtered).cumprod().plot()
(1 + filtered_2).cumprod().plot()
(1 + filtered_3).cumprod().plot()
(1 + df['fwd returns']).cumprod().plot()
results

num_days = (bounded.index[-1] - bounded.index[0]).days
num_years = num_days / 365.25
total_return = (1 + 2 * filtered).cumprod().dropna()
cagr = (total_return.iloc[-1] / total_return.iloc[0]) ** (1 / num_years) - 1
max_dd = drawdown(total_return).min()
print('CAGR: {}'.format(cagr))
print('Max DD: {}'.format(max_dd))

(df['SPY Close'].iloc[-1] / df['SPY Close'].iloc[0]) ** (1 / num_years) - 1

