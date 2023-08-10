import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import datetime
import csv
import os
from sklearn.metrics import r2_score, mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from modules.prediction import load_all_data
from modules.prediction import precrime_train_test_split
from modules.prediction import load_splits
from modules.prediction import create_all_splits
from modules.prediction import sample_model
from modules.prediction_model import sample_model_NB
from modules.prediction_model import sample_model_RF

crime_data = load_all_data()
splits = load_splits()
train_test_data = create_all_splits(crime_data, splits)

X_train_fine, X_test_fine, y_train_fine, y_test_fine = train_test_data['fine']
X_train_coarse, X_test_coarse, y_train_coarse, y_test_coarse = train_test_data['coarse']
X_train_2016, X_test_2016, y_train_2016, y_test_2016 = train_test_data['2016']

X_train_fine

y_train_fine

X_test_fine

y_test_fine

X_train_coarse

y_train_coarse

X_test_coarse

y_test_coarse

X_train_2016

y_train_2016

X_test_2016

y_test_2016

y_pred = sample_model(X_train_fine, y_train_fine, X_test_fine)

y_pred

for crime_type in y_train_fine.select_dtypes(exclude=['object']).columns:
    print('{0}: R2 = {1:.1f}, MSE = {2:.4f}'.format(
            crime_type,
            100*r2_score(y_test_fine[crime_type], y_pred[crime_type]),
            mean_squared_error(y_test_fine[crime_type], y_pred[crime_type]
    )))

os.makedirs('../model_predictions/fine/', exist_ok=True)
y_pred.to_csv('../model_predictions/fine/ridge_regression.csv', quoting=csv.QUOTE_NONNUMERIC)

y_pred_NB = sample_model_NB(X_train_fine, y_train_fine, X_test_fine)

y_pred_NB

for crime_type in y_train_fine.select_dtypes(exclude=['object']).columns:
    print('{0}: R2 = {1:.1f}, MSE = {2:.4f}'.format(
            crime_type,
            100*r2_score(y_test_fine[crime_type], y_pred_NB[crime_type]),
            mean_squared_error(y_test_fine[crime_type], y_pred_NB[crime_type]
    )))

y_pred_RF = sample_model_RF(X_train_fine, y_train_fine, X_test_fine)
for crime_type in y_train_fine.select_dtypes(exclude=['object']).columns:
    print('{0}: R2 = {1:.1f}, MSE = {2:.4f}'.format(
            crime_type,
            100*r2_score(y_test_fine[crime_type], y_pred_RF[crime_type]),
            mean_squared_error(y_test_fine[crime_type], y_pred_RF[crime_type]
    )))

y_pred_NB_coarse = sample_model_NB(X_train_coarse, y_train_coarse, X_test_coarse)
for crime_type in y_train_coarse.select_dtypes(exclude=['object']).columns:
    print('{0}: R2 = {1:.1f}, MSE = {2:.4f}'.format(
            crime_type,
            100*r2_score(y_test_coarse[crime_type], y_pred_NB_coarse[crime_type]),
            mean_squared_error(y_test_coarse[crime_type], y_pred_NB_coarse[crime_type]
    )))

y_pred_NB_2016 = sample_model_NB(X_train_2016, y_train_2016, X_test_2016)
for crime_type in y_train_2016.select_dtypes(exclude=['object']).columns:
    print('{0}: R2 = {1:.1f}, MSE = {2:.4f}'.format(
            crime_type,
            100*r2_score(y_test_2016[crime_type], y_pred_NB_2016[crime_type]),
            mean_squared_error(y_test_2016[crime_type], y_pred_NB_2016[crime_type]
    )))



