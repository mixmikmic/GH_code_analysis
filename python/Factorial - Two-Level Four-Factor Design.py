get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
from numpy.random import rand, seed
import seaborn as sns
import scipy.stats as stats
from matplotlib.pyplot import *

effects = {}
effects[0] = {'x0': 0.0}
effects[1] = {'x1': 0.4,
              'x2': -7.6,
              'x3': 14.1,
              'x4': 66.7}

effects[2] = {'x1-x2': 16.7,
              'x1-x3': 3.1,
              'x1-x4': 5.2,
              'x2-x3': 8.3,
              'x3-x4': 14.3}
effects[3] = {'x1-x2-x3': -0.1,
              'x1-x2-x4': -4.7,
              'x1-x3-x4': 7.7,
              'x2-x3-x4': -2.3}

effects[4] = {'x1-x2-x3-x4': 3.9}

master_dict = {}
for nvars in effects.keys():

    effect = effects[nvars]
    for k in effect.keys():
        v = effect[k]
        master_dict[k] = v

master_df = pd.DataFrame(master_dict,index=['dy']).T
master_df

#print help(master_df.sort)
view = master_df.sort_values(by='dy',ascending=False)
view

# Quantile-quantile plot of effects:

fig = figure(figsize=(4,4))
ax1 = fig.add_subplot(111)

stats.probplot(master_df['dy'], dist="norm", plot=ax1)
ax1.set_title('Effect Size: Quantile-Quantile Plot')
show()



