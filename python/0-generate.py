import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import pandas as pd
print(pd.__version__)

import numpy as np

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs

# https://www.welt.de/motor/news/article156991316/Unfallstatistik-2015.html
# http://www.openculture.com/2017/12/why-incompetent-people-think-theyre-amazing.html
# 0: young drivers with fast cars: red
# 1: reasonable drivers: green
# 2: a little bit older, more kilometers, general noise: yellow
# 3: really old drivers: red
# 4: young drivers: red
# 5: another green just to have a counter part to all the red ones: green
# 6: people who do not drive a lot: green
# 7: people who drive a lot: yellow
# 8: young people with slow cars: yellow

centers = [(200, 35, 50), (160, 50, 25), (170, 55, 30), (170, 75, 20), (170, 30, 30), (190, 45, 40), (160, 40, 15), (180, 50, 45), (140, 25, 15)]
cluster_std = [4, 9, 18, 8, 9, 5, 8, 12, 5]

# X, y = make_blobs(n_samples=300, n_features=3, centers=centers, random_state=13, cluster_std = cluster_std)
# X, y = make_blobs(n_samples=300, n_features=3, centers=centers, random_state=42, cluster_std = cluster_std)
X, y = make_blobs(n_samples=1500, n_features=3, centers=centers, random_state=42, cluster_std = cluster_std)

# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
# X, y = make_classification(n_features=3, n_redundant=0, n_informative=3,
#                              n_clusters_per_class=2, n_classes=3, random_state=42)

feature_names = ['max speed', 'age' ,'thousand km per year']
df = pd.DataFrame(X, columns=feature_names)
df = df.round()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.clip.html

df['max speed'] = df['max speed'].clip(90,400)
df['age'] = df['age'].clip(18,90)
df['thousand km per year'] = df['thousand km per year'].clip(5,500)

# merges clusters into one group
for group in np.nditer(y, op_flags=['readwrite']):
    if group == 3 or group == 4:
        group[...] = 0
    if group == 5 or group == 6:
        group[...] = 1
    if group == 7 or group == 8:
        group[...] = 2

df['group'] = y

df.describe()

# df.to_csv('./insurance-customers-300-2.csv', sep=';', index=False)
# df.to_csv('./insurance-customers-300.csv', sep=';', index=False)
df.to_csv('./insurance-customers-1500.csv', sep=';', index=False)



