import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')

df = pd.read_csv('ExtractedFinalDataset.csv')

bad_features = []
for i in range(8):
    langevin = str(i) + "__max_langevin_fixed_point__m_3__r_30"
    bad_features.append(langevin)
    for j in range(9):
        quantile = (j+1)*0.1
        if quantile != 0.5:
            feature_name = str(i) + "__index_mass_quantile__q_" + str(quantile)
            bad_features.append(feature_name)

df = df.drop(bad_features, axis=1)

df.index = df['9']
df = df.drop(['9'], axis=1)
df['Label'] = "One"
df['Label'][2001.0 <= df.index ] = "Two"
df['Label'][4001.0 <= df.index ] = "Three"
df['Label'][6001.0 <= df.index ] = "Four"
df['Label'][8001.0 <= df.index ] = "Five"
df['Label'][10001.0 <= df.index ] = "Six"

df = df[1:]

df.columns = df.columns.map(lambda t: str(t))
df = df.sort_index(axis=1)

extracted_features = df

subsample = extracted_features.sample(frac=0.05).reset_index(drop=True)
subsample.shape

X = subsample.drop(['Label'], 1)
y = subsample['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

fullset = extracted_features.sample(frac=1).reset_index(drop=True)
X = fullset.drop(['Label'], 1)
y = fullset['Label']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {'n_jobs': [7, 10, 12],
             'n_estimators': [10, 100, 50],
             'min_samples_split': [0.2, 0.5, 0.7, 2]}
model = GridSearchCV(RandomForestClassifier(), param_grid)
model.fit(X_train_full, y_train_full)

print 'Training accuracy:', model.score(np.array(X_train_full),np.array(y_train_full))
print 'Test accuracy:', model.score(np.array(X_test_full), np.array(y_test_full))
print model.best_params_

param_grid = {'n_jobs': [7, 10, 12],
             'n_estimators': [100, 200, 300],
             'min_samples_split': [0.2, 0.5, 0.7, 2]}
model = GridSearchCV(RandomForestClassifier(), param_grid)
model.fit(X_train_full, y_train_full)

print 'Training accuracy:', model.score(np.array(X_train_full),np.array(y_train_full))
print 'Test accuracy:', model.score(np.array(X_test_full), np.array(y_test_full))
print model.best_params_

param_grid = {'n_jobs': [10, 12, 15, 20],
             'n_estimators': [100, 300, 500, 1000]}
model = GridSearchCV(RandomForestClassifier(), param_grid)
model.fit(X_train_full, y_train_full)

print 'Training accuracy:', model.score(np.array(X_train_full),np.array(y_train_full))
print 'Test accuracy:', model.score(np.array(X_test_full), np.array(y_test_full))
print model.best_params_

param_grid = {'n_jobs': [10, 12],
             'n_estimators': [2000, 4000]}
model = GridSearchCV(RandomForestClassifier(), param_grid)
model.fit(X_train_full, y_train_full)

print 'Training accuracy:', model.score(np.array(X_train_full),np.array(y_train_full))
print 'Test accuracy:', model.score(np.array(X_test_full), np.array(y_test_full))
print model.best_params_

from sklearn.externals import joblib
model = RandomForestClassifier(n_jobs = 10, n_estimators = 1000)
model.fit(X, y)

joblib.dump(model, 'bestrandomforest.pkl')

