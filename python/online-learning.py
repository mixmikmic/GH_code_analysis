get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(x, y)
z = 3*np.sin(xx) - 1 + np.sin(xx + 6) + np.cos(yy) + np.cos(yy - 0.5) + 0.6*x
z = 1 - z/20 - 0.5
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, cmap='spring')
ax.set_zlim(0, 1)
ax.set_zlabel('Model Error', fontsize=20, labelpad=10)
ax.set_xlabel('Model Parameter', fontsize=20, labelpad=15)
ax.set_ylabel('Model Parameter', fontsize=20, labelpad=15)
ax.view_init(elev=30., azim=30)

import os
import sys
import requests
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import Bunch


def load_ozone():
    ozone_dir = 'uci_ozone'
    data_dir = datasets.get_data_home()
    data_path = os.path.join(data_dir, ozone_dir, 'onehr.data')
    descr_path = os.path.join(data_dir, ozone_dir, 'onehr.names')
    ozone_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data'
    ozone_descr = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.names'
    os.makedirs(os.path.join(data_dir, ozone_dir), exist_ok=True)
    columns = [
        'WSR0', 'WSR1', 'WSR2', 'WSR3', 'WSR4', 'WSR5', 'WSR6', 'WSR7', 'WSR8', 'WSR9',
        'WSR10', 'WSR11', 'WSR12', 'WSR13', 'WSR14', 'WSR15', 'WSR16', 'WSR17', 'WSR18', 'WSR19',
        'WSR20', 'WSR21', 'WSR22', 'WSR23', 'WSR_PK', 'WSR_AV',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
        'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19',
        'T20', 'T21', 'T22', 'T23', 'T_PK', 'T_AV', 'T85',
        'RH85', 'U85', 'V85', 'HT85', 'T70', 'RH70', 'U70', 'V70',
        'HT70', 'T50', 'RH50', 'U50', 'V50', 'HT50', 'KI', 'TT',
        'SLP', 'SLP_', 'Precp', 'Ozone']
    try:
        with open(descr_path, 'r') as f:
            descr = f.read()
    except IOError:
        print('Downloading file from', ozone_descr, file=sys.stderr)
        r = requests.get(ozone_descr)
        with open(descr_path, 'w') as f:
            f.write(r.text)
        descr = r.text
        r.close()
    try:
        data = pd.read_csv(data_path, delimiter=',',
                           na_values=['?'], names=columns, parse_dates=True)
        data.fillna(data.mean(), inplace=True)
    except IOError:
        print('Downloading file from', ozone_data, file=sys.stderr)
        r = requests.get(ozone_data)
        with open(data_path, 'w') as f:
            f.write(r.text)
        r.close()
        data = np.loadtxt(data_path, delimiter=',')
    return Bunch(DESCR=descr,
                 data=data.values[:, :72],
                 feature_names=columns[:72],
                 target=data.values[:, 72],
                 target_names=['normal day', 'ozone day'])


ozone = load_ozone()
print(ozone.DESCR)

xtrain, xtest, ytrain, ytest = train_test_split(
    ozone.data, ozone.target, test_size=0.2, random_state=42)

model = make_pipeline(
    StandardScaler(),
    PCA(n_components=20),
    SGDClassifier(loss='log', penalty='l1',
                  max_iter=500, alpha=0.001, tol=0.01, class_weight={0: 1, 1: 25}))
param_grid = {
    'sgdclassifier__alpha': [0.001, 0.01, 0.1],
    'sgdclassifier__tol': [0.001, 0.01, 0.1],
}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(xtrain, ytrain)
grid.best_score_

grid.best_estimator_

yfit = grid.best_estimator_.predict(xtest)
print(classification_report(ytest, yfit, target_names=ozone.target_names))

ozone1, ozone2, yozone1, yozone2 = train_test_split(
    ozone.data, ozone.target, test_size=0.5, random_state=42)

xtrain1, xtest1, ytrain1, ytest1 = train_test_split(
    ozone1, yozone1, test_size=0.2, random_state=0)
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    SGDClassifier(loss='hinge', penalty='l1', max_iter=200, alpha=0.001,
                  tol=0.001, warm_start=True, class_weight={0: 1, 1: 25}))
model.fit(xtrain1, ytrain1)
yfit = model.predict(xtest1)
print(classification_report(ytest1, yfit, target_names=ozone.target_names))

xtrain2, xtest2, ytrain2, ytest2 = train_test_split(
    ozone2, yozone2, test_size=0.2, random_state=0)
yfit = model.predict(xtest2)
print(classification_report(ytest2, yfit, target_names=ozone.target_names))

model.fit(xtrain2, ytrain2)
yfit = model.predict(xtest2)
print(classification_report(ytest2, yfit, target_names=ozone.target_names))

