get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import cartopy.crs as ccrs
from ipywidgets import interact, SelectionSlider
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Edit this list of variables to load in a smaller subset.
data_vars = ["TS", "PSL", "TMQ", "U_500", "V_500"]

def read_data(path, f_prefix, data_vars):
    X_coll = []
    for data_var in data_vars:
        nc_file = join(path, "data", f_prefix + "_{0}.nc".format(data_var))
        print(nc_file)
        ds = xr.open_dataset(nc_file, decode_times=False)
        ds.load()
        X_coll.append(ds[data_var].stack(enstime=("ens", "time")).transpose("enstime", "lat", "lon"))
        ds.close()
    X_ds = xr.merge(X_coll)
    y = pd.read_csv(join(path, "data", f_prefix + "_precip_90.csv"), index_col="Year")
    y_array = np.concatenate([y[c] for c in y.columns])
    return X_ds, y_array

train_X, train_y = read_data("./", "train", data_vars)

train_X["TS"].sel(ens=0, time=334.0).shape

years = np.arange(850, 2005)
def plot_grid(ens, year, var):
    ti = np.where(year == years)[0][0]
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines()
    min_val = train_X[var].min()
    max_val = train_X[var].max()
    cont = ax.contourf(train_X["lon"] - 180, train_X["lat"], 
                       train_X[var].sel(ens=ens, time=train_X["time"].values[ti]),
                       np.linspace(min_val, max_val, 20))
    ax.set_title(var + " " + "Year: {0:d} Member {1}".format(year, ens))
    plt.colorbar(cont)
interact(plot_grid, ens=[0, 1, 2, 3], year=SelectionSlider(options=years.tolist()), 
         var=data_vars)

train_X_anomalies = xr.merge([(train_X[var] - train_X[var].mean(axis=0)) / (train_X[var].std(axis=0)) for var in data_vars])

def plot_anomaly(ens, year, var):
    ti = np.where(year == years)[0][0]
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines()
    min_val = -5
    max_val = 5
    cont = ax.contourf(train_X_anomalies["lon"] - 180, train_X_anomalies["lat"], 
                       train_X_anomalies[var].sel(ens=ens, time=train_X_anomalies["time"].values[ti]),
                       np.linspace(min_val, max_val, 11), cmap="RdBu_r")
    ax.set_title(var + " " + "Year: {0:d} Member {1}".format(year, ens))
    plt.colorbar(cont)
interact(plot_anomaly, ens=[0, 1, 2, 3], year=SelectionSlider(options=years.tolist()), 
         var=data_vars)

rain_data = pd.read_csv("data/train_precip.csv", index_col="Year")
rain_data.rolling(25).mean().plot(figsize=(15, 5))
plt.ylabel("DJF Precip")

rain_data.hist(bins=np.arange(0, 1600, 100), figsize=(10, 5))

lags = np.arange(1, 20)
autocorr = np.zeros((rain_data.columns.size, lags.size))
plt.figure(figsize=(8, 5))
for c, col in enumerate(rain_data.columns):
    autocorr[c] = np.array([rain_data[col].autocorr(l) for l in range(1, 20)])
    plt.plot(lags, np.abs(autocorr[c]), label=col)
plt.xticks(lags)
plt.title("California Precip Temporal Autocorrelation")
plt.xlabel("Lag (Years)")
plt.legend(loc=0)

class FeatureExtractor():
    def __init__(self):
        self.means = {}
        self.sds = {}
        self.variables = ["TS", "PSL", "TMQ"]
        self.pca = {}
        self.num_comps = 20

    def fit(self, X_ds, y):
        for var in self.variables:
            if var not in self.means.keys():
                self.means[var] = X_ds[var].mean(axis=0).values.astype(np.float32)
                self.sds[var] = X_ds[var].std(axis=0).values.astype(np.float32)
                self.sds[var][self.sds[var] == 0] = 1
            var_norm = (X_ds[var] - self.means[var]) / self.sds[var]
            var_flat = var_norm.stack(latlon=("lat", "lon")).values
            del var_norm
            var_flat[np.isnan(var_flat)] = 0
            self.pca[var] = PCA(n_components=self.num_comps)
            self.pca[var].fit(var_flat)
            del var_flat

    def transform(self, X_ds):
        X = np.zeros((np.prod(X_ds[self.variables[0]].shape[:1]), 
                      self.num_comps * len(self.variables)), dtype=np.float32)
        c = 0
        for var in self.variables:
            var_norm = (X_ds[var] - self.means[var]) / self.sds[var]
            var_flat = var_norm.stack(latlon=("lat", "lon")).values
            del var_norm
            var_flat[np.isnan(var_flat)] = 0
            X[:, c:c+self.num_comps] = self.pca[var].transform(var_flat)
            c += self.num_comps
            del var_flat
        return X

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegression(C=0.01, penalty="l1")

    def fit(self, X, y): 
        self.clf.fit(X, y)

    def predict_proba(self, X): 
        return self.clf.predict_proba(X)

fe = FeatureExtractor()
fe.fit(train_X, train_y)
X = fe.transform(train_X)
cls = Classifier()
cls.fit(X, train_y)

coefs = cls.clf.coef_[0]
coef_rankings = np.argsort(np.abs(coefs))[::-1]
fig, axes = plt.subplots(3, 3, figsize=(16, 9), 
                         subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
axef = axes.ravel()
for c, coef_rank in enumerate(coef_rankings[:9]):
    c_var = data_vars[int(np.floor(coef_rank / fe.num_comps))]
    c_comp = coef_rank % fe.num_comps
    comp_vals = fe.pca[c_var].components_[c_comp]
    axef[c].coastlines()
    axef[c].contourf(train_X["lon"] - 180, 
                     train_X["lat"], 
                     fe.pca[c_var].components_[c_comp].reshape(train_X[c_var].shape[1:]),
                     np.linspace(-0.04, 0.04, 11), cmap="RdBu_r")
    axef[c].set_title("{0} Comp {1:d} Coef: {2:0.4f}".format(c_var, c_comp, coefs[coef_rank]))

get_ipython().system('source activate ramp_ci_2017; ramp_test_submission')

import imp
problem = imp.load_source('', 'problem.py')

X_train, y_train = problem.get_train_data()

train_is, test_is = list(problem.get_cv(X_train, y_train))[0]
print(len(train_is), len(test_is))

ts_fe, reg = problem.workflow.train_submission(
    'submissions/starting_kit', X_train, y_train, train_is)

y_pred = problem.workflow.test_submission((ts_fe, reg), X_train)

score_function = problem.score_types[0]

score_train = score_function(y_train[train_is], y_pred[train_is][:, 1])
print(score_train)

score_valid = score_function(y_train[test_is], y_pred[test_is][:, 1])

X_test, y_test = problem.get_test_data(path="./")

y_test_pred = problem.workflow.test_submission((ts_fe, reg), X_test)

score_test = score_function(y_test, y_test_pred[:, 1])
print(score_test)



