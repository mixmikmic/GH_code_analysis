import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

data = pd.read_csv('../../recombination/rspr/rspr.csv', header=None, index_col=None)
rename = {i: ii for (i, ii) in enumerate(range(1,13))}
data.rename(columns=rename, index=rename, inplace=True)
data = data.replace(0,np.nan)
mean = data.stack().mean()

fig, ax = plt.subplots(figsize=(5.2, 4.3))
ax = sns.heatmap(data, annot=True, cmap = plt.cm.GnBu, ax=ax, square=True)

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=10)
for label in ax.get_yticklabels() + ax.get_xticklabels() + ax.texts :
        label.set_size(10)

ax.set_xlabel('Segment Number', fontsize=12)
ax.set_ylabel('Segment Number', fontsize=12)
ax.set_title('')
plt.tight_layout()

plt.savefig('../png/FigS2.png', dpi=300, bbox_inches='tight')
plt.show()



