import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

app = pd.read_pickle('/Users/krystal/Desktop/app_clean.p')
app.head()

app = app.drop_duplicates()

app['overall reviews'] = map(lambda x: int(x) if x!='' else np.nan, app['overall reviews'])
app['overall rating'] = map(lambda x: float(x) if x!='' else np.nan, app['overall rating'])
app['current rating'] = map(lambda x: float(x) if x!='' else np.nan, app['current rating'])

multi_language = app.loc[app['multiple languages'] == 'Y']
sin_language = app.loc[app['multiple languages'] == 'N']
multi_language['overall rating'].plot(kind = "density")
sin_language['overall rating'].plot(kind = "density")
plt.xlabel('Overall Rating')
plt.legend(labels = ['multiple languages','single language'], loc='upper right')
plt.title('Distribution of overall rating among apps with multiple/single languages')
plt.show()

import scipy.stats

multi_language = list(multi_language['overall rating'])
sin_language = list(sin_language['overall rating'])

multiple = []
single = []
for each in multi_language:
    if each > 0:
        multiple.append(each)
for each in sin_language:
    if each > 0:
        single.append(each)

print(np.mean(multiple))
print(np.mean(single))

scipy.stats.ttest_ind(multiple, single, equal_var = False)

scipy.stats.f_oneway(multiple, single)

scipy.stats.kruskal(multiple, single)



