import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor, GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import os
import IPython.display as dis

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(8)
path = os.getcwd()

pd.options.display.max_columns = 100

data = pd.read_csv(os.path.join(path, "data", "train.txt"), sep=' ')

data.unit=data.unit.astype(int)

data.describe()

data.dtypes

len(np.unique(data['unit']))

data['s4'].plot.hist()

data[data['unit']==1]['s4'].plot.hist()

data[data['unit']==10]['s4'].plot.hist()

data['s19'].plot()

data[data['unit']==6].reset_index()['RUL'].plot()

import seaborn as sns
corr = data.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corr, 
            xticklabels=data.columns.values,
            yticklabels=data.columns.values,annot=True)

get_ipython().run_cell_magic('file', 'submissions/starting_kit/feature_extractor.py', '\nclass FeatureExtractor(object):\n    def __init__(self):\n        pass\n\n    def fit(self, X_df, y_array):\n        pass\n\n    def transform(self, X_df):\n        return X_df.drop("unit",axis=1)')

get_ipython().run_cell_magic('file', 'submissions/starting_kit/regressor.py', 'from __future__ import absolute_import\nfrom sklearn.base import BaseEstimator\nfrom sklearn.ensemble import RandomForestRegressor\n\n\nclass Regressor(BaseEstimator):\n    \n    def __init__(self):\n        self.reg = RandomForestRegressor()\n\n    def fit(self, X, y):\n        return self.reg.fit(X, y)\n\n    def predict(self, X):\n        return self.reg.predict(X)')

get_ipython().system('ramp_test_submission ')

