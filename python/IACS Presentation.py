import numpy as np
import pandas as pd
X_returns = np.random.normal(0, 1, 100) # Generate the daily returns
# sum them and shift all the prices up into a reasonable range
X = pd.Series(np.cumsum(X_returns), name='X') + 50
X.plot()

some_noise = np.random.normal(0, 1, 100)
Y = X + 5 + some_noise
Y.name = 'Y'
pd.concat([X, Y], axis=1).plot()

import statsmodels
from statsmodels.tsa.stattools import coint

score, pvalue, _ = coint(X, Y)
print pvalue
print pvalue < 0.05

def find_cointegrated_pairs(securities_panel):
    n = len(securities_panel.minor_axis)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = securities_panel.keys
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = securities_panel.minor_xs(securities_panel.minor_axis[i])
            S2 = securities_panel.minor_xs(securities_panel.minor_axis[j])
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((securities_panel.minor_axis[i], securities_panel.minor_axis[j]))
    return score_matrix, pvalue_matrix, pairs

symbol_list = ['ABGB', 'ASTI', 'CSUN', 'DQ', 'FSLR', 'HK', 'AAPL', 'MSFT']
securities_panel = get_pricing(symbol_list, fields=['price'], start_date='2014-01-01', end_date='2015-01-01')
securities_panel.minor_axis = map(lambda x: x.symbol, securities_panel.minor_axis)

# series_list = load_prices(symbol_list)
scores, pvalues, pairs = find_cointegrated_pairs(securities_panel)
import seaborn
seaborn.heatmap(pvalues, xticklabels=symbol_list, yticklabels=symbol_list, mask = (pvalues >= 0.5))
print pairs

S = securities_panel.loc['price', :, ['ABGB', 'FSLR']]
S.plot()

# Get the difference
difference = securities_panel.loc['price', :, 'ABGB'] - securities_panel.loc['price', :, 'FSLR']
difference.name = 'diff'
# Get the 5 day moving average of the difference
diff_mavg5 = pd.rolling_mean(difference, window=5)
diff_mavg5.name = 'diff 5d mavg'
# Get the 60 day moving average
diff_mavg60 = pd.rolling_mean(difference, window=60)
diff_mavg60.name = 'diff 60d mavg'

pd.concat([diff_mavg60, diff_mavg5, difference], axis=1).plot()

# Take a rolling 60 day standard deviation
std_60 = pd.rolling_std(difference, window=60)
std_60.name = 'std 60d'

# Compute the z score at each time
zscore = (diff_mavg5 - diff_mavg60)/std_60
zscore.name = 'z-score'
zscore.plot()

# Plot the prices scaled down along with the negative z-score
pd.concat([S/10, -zscore], axis=1).plot()



