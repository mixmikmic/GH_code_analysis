import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

app = pd.read_pickle('/Users/krystal/Desktop/app_cleaned.pickle')
app.head()

app = app.drop_duplicates()

app.dtypes.index

data_q3 = app[np.isfinite(app['current_rating']) & np.isfinite(app['is_InAppPurcased'])]

data_q3['is_InAppPurcased'].value_counts()

free = data_q3.loc[data_q3['is_InAppPurcased'] == 0]
paid = data_q3.loc[data_q3['is_InAppPurcased'] == 1]
free['current_rating'].plot(kind = "density")
paid['current_rating'].plot(kind = "density")
plt.xlabel('Current Rating')
plt.legend(labels = ['free','paid'], loc='upper right')
plt.title('Distribution of current rating among free/paid apps')
plt.show()

import scipy.stats

free = list(free['current_rating'])
paid = list(paid['current_rating'])

print(np.mean(free))
print(np.mean(paid))

scipy.stats.ttest_ind(free, paid, equal_var = False)

scipy.stats.f_oneway(free, paid)

scipy.stats.kruskal(free, paid)



