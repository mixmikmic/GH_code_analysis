get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
mpl.style.use('mitch-exp')

import exc_analysis.plotting as excplt

ls

data = pd.read_csv('auto_tustin.csv')
data.set_index('Total Time', inplace=True)
data.head()

dfs = []

for i, idx in enumerate(np.where(data['Traj Time'] < 0.05)[0][:-1]):
    dfs.append(data.iloc[idx: np.where(data['Traj Time'] < 0.05)[0][i + 1]])

dfs[1].head()

tr = 5
fig = plt.figure(figsize=(10, 8))

for i, lbl in enumerate(excplt.labels):
    ax = plt.subplot(4, 2, 2 * i + 1)
    dfs[tr][[lbl + ' Ms']].rolling(window=5, min_periods=3).mean().plot(ax=ax)
    dfs[tr][[lbl + ' Ref']].plot(ax=ax, linestyle='--')
    if i != 3:
        ax.xaxis.label.set_visible(False)
    
    ax = plt.subplot(4, 2, (i + 1) * 2)
    dfs[tr][[lbl + ' Cmd']].plot(ax=ax)
    if i != 3:
        ax.xaxis.label.set_visible(False)

fig.text(0, 0.65, 'Actuator Position (cm)', fontsize=18, verticalalignment='center', rotation='vertical')
fig.text(0, 0.15, 'Angle (rad)', fontsize=18, verticalalignment='center', rotation='vertical')
plt.tight_layout()
plt.savefig('figs/control_tustin_ziegler.pdf')



