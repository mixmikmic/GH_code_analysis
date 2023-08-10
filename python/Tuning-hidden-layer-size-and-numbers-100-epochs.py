import pandas as pd

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('./models_100_epochs/test_mse.csv', sep='\t', header=None, names=['hln', 'hls', 'train_mse', 'test_mse'])

df.sort_values('test_mse').head()

df.shape

train_mse = df.pivot(index='hln', columns='hls', values='train_mse')

plt.figure(figsize=(8, 6))
plt.imshow(train_mse, vmin=0, vmax=1)
plt.colorbar()
plt.xlabel('hidden_layer_size')
plt.ylabel('hidden_layer_num')

test_mse = df.pivot(index='hln', columns='hls', values='test_mse')

plt.figure(figsize=(8, 6))
plt.imshow(test_mse)
plt.colorbar()
plt.xlabel('hidden_layer_size')
plt.ylabel('hidden_layer_num')

ax = plt.axes()
for k, grp in df.groupby('hln'):
    grp.sort_values('hls').plot(x='hls', y='train_mse', 
                                label='hidden layer number: {0}'.format(grp.hln.unique()[0]), ax=ax)
ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.set_xlabel('hidden layer size')
ax.set_ylabel('Mean squared error')



