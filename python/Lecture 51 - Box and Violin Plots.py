import numpy as np
from numpy.random import randn
import pandas as pd

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot 
import seaborn as sns
get_ipython().magic('matplotlib inline')

data1=randn(100)
data2=randn(100)

sns.boxplot(data1)
sns.boxplot(data2)

sns.boxplot([data1,data2],whis=np.inf)

sns.boxplot([data1,data2],vert=True)

#Normal distribution
data1=stats.norm(0,5).rvs(100)

#Two gamma distributions. Concatinated together
data2=np.concatenate([stats.gamma(5).rvs(50)-1,
                     stats.gamma(5).rvs(50)*-1])
sns.boxplot([data1,data2],whis=np.inf)

sns.violinplot([data1,data2])

sns.violinplot(data2,bw=0.01)

sns.violinplot(data1,inner='stick')



