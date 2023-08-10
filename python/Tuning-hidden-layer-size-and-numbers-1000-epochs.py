import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./models_1000_epochs/test_mse.csv', sep='\t', header=None, names=['hln', 'hls', 'train_mse', 'test_mse'])

df.sort_values('test_mse').head()

df.shape

# highlight best configuration
best_idx = df.sort_values('test_mse').index[0]
ndf = df.copy()
ndf.loc[best_idx, 'train_mse'] = 1
ndf.loc[best_idx, 'test_mse'] = 1

fig, axes = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={
    "width_ratios":[1, 1, 0.05]
})
axes = axes.ravel()

for k, i in enumerate(['train_mse', 'test_mse']):
    ax = axes[k]
    mse = ndf.pivot(index='hln', columns='hls', values=i)
    # good color map distinguishes the difference at the lower right corner, 
    # which show that 1 layer and two layer do make a difference!
    im = ax.imshow(mse, vmin=0, vmax=1, cmap='gist_stern')
    ax.set_xlabel('hidden_layer_size')
    ax.set_ylabel('hidden_layer_num')
    ax.set_title(i)
    ax.set_xticks(np.arange(mse.shape[1]))
    ax.set_xticklabels(mse.columns.values)
    ax.set_yticks(np.arange(mse.shape[0]))
    ax.set_yticklabels(mse.index.values)
    ax.invert_yaxis()

fig.colorbar(im, cax=axes[-1])
plt.tight_layout()

# The white spot at low x axis is due a missing data point
df.query('hln == 9').query('hls == 3')

ax = plt.axes()
for k, grp in df.groupby('hln'):
    grp.sort_values('hls').plot(x='hls', y='train_mse', 
                                label='hidden layer number: {0}'.format(grp.hln.unique()[0]), ax=ax)
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.set_xlabel('hidden layer size')
ax.set_ylabel('Train mean squared error')

ax = plt.axes()
for k, grp in df.query('hln < 6').groupby('hln'):
    grp.sort_values('hls').plot(x='hls', y='train_mse', 
                                label='hidden layer number: {0}'.format(grp.hln.unique()[0]), ax=ax)
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.set_xlabel('hidden layer size')
ax.set_ylabel('Train mean squared error')



