import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

random_state = 42

data_to_use = 100000

types = {'row_id': np.dtype(int),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(int),
         'place_id': np.dtype(int) }

data = pd.read_csv('../data/train_fb.csv',
                   index_col='row_id', dtype=types)[:data_to_use]

data.head()

data.describe()

# (29118021, 5) for full train dataset
data.shape

# Looks almost uniform distributes :(
plt.figure(figsize=(12, 12))
plt.scatter(data['x'], data['y'], c=data['place_id'], cmap='Set1', alpha=0.7)
plt.show()

# 108390 categories for full dataset
print('Number of categiries: {0}'.format(len(data['place_id'].unique())))

counted_values = data['place_id'].value_counts()

num = 0
for key, val in counted_values.iteritems():
    if val < 1000:
        break
    num += 1
        
print('Number of categories that occures more than 1000 times: {0}').format(num)

nrows = 10000
ids = [8523065625, 1757726713, 1137537235, 6567393236, 7440663949]
colors = ['red', 'green', 'blue', 'orange', 'purple']

for i, c in zip(ids, colors):
    plot_data = data[data['place_id'] == i]
    plt.scatter(plot_data['x'][:nrows], plot_data['y'][:nrows],
                color=c, alpha=0.4)

nrows = 10000
ids = [8523065625, 1757726713, 1137537235, 6567393236, 7440663949]
colors = ['red', 'green', 'blue', 'orange', 'purple']

for i, c in zip(ids, colors):
    plot_data = data[data['place_id'] == i]
    plt.scatter(plot_data['x'][:nrows], plot_data['time'][:nrows],
                color=c, alpha=0.4)

from sklearn.cross_validation import train_test_split

X = data[['x', 'y']]
y = data['place_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state)

# from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# pred = clf.predict(X_test)
# accuracy_score(y_true, pred)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
accuracy_score(y_true, pred)

