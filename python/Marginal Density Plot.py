get_ipython().magic('pylab inline')
import pysd
import numpy as np
import pandas as pd
import seaborn

model = pysd.read_vensim('../../models/Climate/Atmospheric_Bathtub.mdl')
print model.doc()

n_runs = 1000
runs = pd.DataFrame({'Emissions': np.random.exponential(scale=10000, size=n_runs)})
runs.head()

result = runs.apply(lambda p: model.run(params=dict(p))['Excess Atmospheric Carbon'],
                    axis=1).T
result.head()

# left side should have all traces plotted
plt.subplot2grid((1,4), loc=(0,0), colspan=3)
[plt.plot(result.index, result[i], 'b', alpha=.02) for i in result.columns]
plt.ylim(0, max(result.iloc[-1]))

# right side has gaussian KDE on last timestamp
plt.subplot2grid((1,4), loc=(0,3))
seaborn.kdeplot(result.iloc[-1], vertical=True)
plt.ylim(0, max(result.iloc[-1]));
plt.yticks([])
plt.xticks([])

plt.suptitle('Emissions scenarios under uncertainty', fontsize=16);

