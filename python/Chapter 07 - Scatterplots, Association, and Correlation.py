import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_style('whitegrid')

derby = pd.read_table("./data/Kentucky_Derby_2011.txt")

plt.scatter(derby.Year, derby.speed);
plt.xlabel("Year")
plt.ylabel("Speed")
plt.title("Kentuck Derby : Year vs. Speed");

# calc with numpy
np.corrcoef(derby.Year, derby.speed)

# calc with scipy
import scipy.stats as stats
stats.pearsonr(derby.Year, derby.speed)

# calc with pandas
derby.corr()

stats.kendalltau([12, 2, 1, 12, 2], [1, 4, 7, 1, 0])

stats.spearmanr([1,2,3,4,5],[5,6,7,8,7])

