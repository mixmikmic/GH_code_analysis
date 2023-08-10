import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
get_ipython().magic('matplotlib inline')

time_train_full = pd.read_csv("train_2_final.csv")
time_test = time_train_full.iloc[:, -62:].assign(Page = time_train_full.loc[:, "Page"])
time_train = time_train_full.iloc[:, -72:-62].assign(Page = time_train_full.loc[:, "Page"])
time_train.head()

time_train_median = time_train.median(axis = 1, skipna = True).fillna(0)
time_test_melt = pd.melt(time_test, id_vars = ["Page"])
time_train_median_frame = time_train.loc[:, "Page"].to_frame(name = "Page").assign(page_median = time_train_median)
time_test_melt_join = time_test_melt.merge(time_train_median_frame, on = "Page", how = "left")
page_median_log = time_test_melt_join.loc[:, "page_median"]
page_median_log[page_median_log == 0] = 1
page_median_log = np.log(page_median_log.values)
page_observed_log = time_test_melt_join.loc[:, "value"]
page_observed_log[page_observed_log == 0] = 1
page_observed_log = np.log(page_observed_log.values)
plt.scatter(page_median_log, page_observed_log)

prophet_final = pd.read_csv("kaggle_time_train_prophet_results_all_series_train.csv").dropna()
prophet_final.loc[prophet_final["yhat"] < 0, "yhat"] = 0
prophet_final.head()

plt.scatter(np.log(prophet_final["yhat"] + 1), np.log(prophet_final["observed_y"] + 1))

plt.scatter(np.log(prophet_final["y_median"] + 1), np.log(prophet_final["observed_y"] + 1))

plt.scatter(np.log(prophet_final["yhat"] + 1), np.log(prophet_final["y_median"] + 1))

def time_predict(weights):
    lin_predictor = weights[0] * prophet_final["yhat"].values + weights[1] * prophet_final["y_median"].values + weights[2]
    time_prediction = np.rint(lin_predictor)
    return(time_prediction)

# SMAPE score. Adapts code from https://www.kaggle.com/cpmpml/smape-weirdness
def smape_loss(weights):
    y_pred = time_predict(weights)
    y_true = prophet_final["observed_y"].values
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return(np.nanmean(diff))

res_lin = minimize(smape_loss, np.array([1.0, 1.0, 0.0]), method = "Nelder-Mead", tol=1e-6)
res_lin["x"]

def time_predict(weights):
    lin_predictor = weights[0] * prophet_final["yhat"].values  + weights[1]
    time_prediction = np.rint(lin_predictor)
    return(time_prediction)

# SMAPE score. Adapts code from https://www.kaggle.com/cpmpml/smape-weirdness
def smape_loss(weights):
    y_pred = time_predict(weights)
    y_true = prophet_final["observed_y"].values
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return(np.nanmean(diff))

res_lin_2 = minimize(smape_loss, np.array([1.0, 0.0]), method = "Nelder-Mead", tol=1e-6)
res_lin_2["x"]

