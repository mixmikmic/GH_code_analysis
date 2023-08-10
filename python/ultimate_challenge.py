import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import permutation_test_score
get_ipython().magic('matplotlib inline')

logins = pd.read_json("logins.json")
logins["count"] = 1
logins.head()

login_counts = logins.groupby(pd.Grouper(key='login_time', freq='900s')).count()
login_counts['day_of_week'] = (pd.DatetimeIndex(login_counts.index).dayofweek)
login_counts['hour'] = (pd.DatetimeIndex(login_counts.index).hour)
login_counts.head()

plt.plot_date(date2num(list(login_counts.index)), login_counts["count"].values)

login_counts["day_of_week"][login_counts["count"] > 20].value_counts()

login_counts["hour"][login_counts["count"] > 20].value_counts()

with open("ultimate_data_challenge.json") as f:
    pred_data = json.load(f)

pred_data = pd.DataFrame(pred_data)
pred_data.head()

min(pd.to_datetime(pred_data["last_trip_date"]))

max(pd.to_datetime(pred_data["last_trip_date"]))

active_ind = (pd.to_datetime(pred_data["last_trip_date"]) < pd.to_datetime(max(pred_data["last_trip_date"])) - pd.Timedelta("30 days")) + 0
active_ind.value_counts()

active_ind.value_counts()[0]/len(active_ind)

min(pd.to_datetime(pred_data["signup_date"]))

max(pd.to_datetime(pred_data["signup_date"]))

pred_features = pred_data.iloc[:,[0,1,2,3,4,6,8,9,10,11]]
pred_features.head()

pred_features.city.value_counts()

pred_features.phone.value_counts()

pred_features.loc[:,"ultimate_black_user"] = pred_features.loc[:,"ultimate_black_user"] + 0
pred_features.loc[:,"phone"] = (pred_features.loc[:,"phone"] == "iPhone") + 0
pred_features_city = pred_features.loc[:,"city"]
pred_features.loc[:,"city"] = (pred_features.loc[:,"city"] == "Winterfell") + 0
pred_features.loc[:,"city_2"] = (pred_features_city == "Astapor") + 0

pred_features.head()

pred_features = pred_features.fillna(-1)

rf = RandomForestClassifier(max_depth = 5)
score, permutation_scores, pvalue = permutation_test_score(rf, pred_features.values, active_ind.values, scoring="accuracy", cv = 5, n_permutations = 100, n_jobs=1)

permutation_scores

score

pvalue

active_ind.value_counts()[1]/len(active_ind)

