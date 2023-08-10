import process
df = process.processed_data()

get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd

colors = df.loc[:,['colorCode', 'returnQuantity']]
colors['10'] = list(pd.cut(colors['colorCode'], range(0, 10001, 10)))
colors['100'] = list(pd.cut(colors['colorCode'], range(0, 10001, 100)))
colors['1000'] = list(pd.cut(colors['colorCode'], range(0, 10001, 1000)))

import matplotlib.pyplot as plt

evaluation = dmc.evaluation.features(colors)

fig, axes = plt.subplots(2,1)
fig.set_size_inches([16, 9])
evaluation.loc['1000', ['retProb']].plot(kind='bar', ax=axes[0], ylim=[0, 1])
evaluation.loc['1000', ['count']].plot(kind='bar', ax=axes[1])

fig, axes = plt.subplots(2,1)
fig.set_size_inches([16, 9])
evaluation.loc['100', ['retProb']].plot(kind='bar', ax=axes[0], ylim=[0, 1])
evaluation.loc['100', ['count']].plot(kind='bar', ax=axes[1])

fig, axes = plt.subplots(2,1)
fig.set_size_inches([16, 9])
evaluation.loc['10', ['retProb']].plot(kind='bar', ax=axes[0], ylim=[0, 1])
evaluation.loc['10', ['count']].plot(kind='bar', ax=axes[1])

mean = evaluation.loc['colorCode'].avgRet.mean()
std = evaluation.loc['colorCode'].avgRet.std()
print('mean: ', mean)
print('std: ', std)

split = '1000'
mean_distances = evaluation.loc[split,'retProb'].sub(mean).abs()
outliers = evaluation.loc[split].loc[mean_distances > std]['count']

print(outliers.sum())
outliers.sum() / len(colors)

split = '100'
mean_distances = evaluation.loc[split,'retProb'].sub(mean).abs()
outliers = evaluation.loc[split].loc[mean_distances > std]['count']

print(outliers.sum())
outliers.sum() / len(colors)

split = '10'
mean_distances = evaluation.loc[split,'retProb'].sub(mean).abs()
outliers = evaluation.loc[split].loc[mean_distances > std]['count']

print(outliers.sum())
outliers.sum() / len(colors)

split = 'colorCode'
mean_distances = evaluation.loc[split,'retProb'].sub(mean).abs()
outliers = evaluation.loc[split].loc[mean_distances > std]['count']

print(outliers.sum())
outliers.sum() / len(colors)

import dmc

colors['binnedColorCode'] = dmc.preprocessing.binned_color_code(colors)
evaluation = dmc.evaluation.features(colors.loc[:,['binnedColorCode', 'returnQuantity']])

fig, axes = plt.subplots(3,1)
fig.set_size_inches([16, 9])
evaluation.loc['binnedColorCode', ['retProb']].plot(kind='bar', ax=axes[0], ylim=[0, 1])
evaluation.loc['binnedColorCode', ['count']].plot(kind='bar', ax=axes[1])
evaluation.loc['binnedColorCode', ['retProb']].plot(kind='hist', bins=100, ax=axes[2])

split = 'binnedColorCode'
mean_distances = evaluation.loc[split,'retProb'].sub(mean).abs()
outliers = evaluation.loc[split].loc[mean_distances > std]['count']

print(outliers.sum())
outliers.sum() / len(colors)

