# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('clean_data.csv', delimiter=',')
data.replace(to_replace='blues', value=0, inplace=True)
data.replace(to_replace='classical', value=1, inplace=True)
data.replace(to_replace='country', value=2, inplace=True)
data.replace(to_replace='disco', value=3, inplace=True)
data.replace(to_replace='hiphop', value=4, inplace=True)
data.replace(to_replace='jazz', value=5, inplace=True)
data.replace(to_replace='metal', value=6, inplace=True)
data.replace(to_replace='pop', value=7, inplace=True)
data.replace(to_replace='reggae', value=8, inplace=True)
data.replace(to_replace='rock', value=9, inplace=True)
#     data = data.sample(frac=1)
x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1:]
print(x_data, y_data)

print(data.columns.values)

data.head()

data.tail()

data.info()

data.describe()

g = sns.FacetGrid(data, col='28')
for i in range(0,1):
    plt.figure(i)
    g.map(plt.hist, str(i))

import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1:]

from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(x_data, y_data)
feature_importances = model.feature_importances_
# for i in range(len(feature_importances)):
#     print(i+1, feature_importances[i])
# sorted_array = np.sort(feature_importances)
for i in range(len(feature_importances)):
    print(i, feature_importances[i])

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
fit = pca.fit(x_data)
# summarize components
print(f'Explained Variance:{fit.explained_variance_ratio_}')
print(fit.components_)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 27)
fit = rfe.fit(x_data,y_data)
print(f'Num Features: {fit.n_features_}')
print(f'Selected Features: {fit.support_}')
print(f'Feature Ranking: {fit.ranking_}')

new_data2 = x_data.iloc[:,fit.support_]

new_data2

new_data2['28']=y_data
new_data2.to_csv('new_data.csv')



