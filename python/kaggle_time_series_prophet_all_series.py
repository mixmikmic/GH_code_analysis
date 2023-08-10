import pandas as pd
import numpy as np
from fbprophet import Prophet

def estimate_prophet(row_index, future_periods = 62):
    test_row_name = time_train.iloc[row_index, 0]
    test_row = time_train.iloc[row_index, 1:-future_periods]
    future_row = time_train.iloc[row_index, -future_periods:]
    test_row_df = pd.DataFrame(index=range(0, len(test_row)))
    test_row_df['ds'] = pd.to_datetime(test_row.index)
    test_row_df['y'] = test_row.values.astype("float")
    prophet_model = Prophet(yearly_seasonality = False)
    prophet_model.fit(test_row_df)
    test_row_future = prophet_model.make_future_dataframe(periods = future_periods)
    test_row_forecast = prophet_model.predict(test_row_future.iloc[-future_periods:, :])
    return(test_row_forecast.iloc[:, [0,-1]].assign(series_name = test_row_name).assign(observed_y = future_row.values).assign(y_median = time_train.iloc[row_index, -(future_periods + 30):-future_periods].median()))

def estimate_prophet_error(row_index, future_periods = 62):
    test_row_name = time_train.iloc[row_index, 0]
    test_row = time_train.iloc[row_index, 1:-future_periods]
    future_row = time_train.iloc[row_index, -future_periods:]
    dummy_row = time_train.iloc[0, 1:-future_periods]
    test_row_df = pd.DataFrame(index=range(0, len(dummy_row)))
    test_row_df['ds'] = pd.to_datetime(dummy_row.index)
    test_row_df['y'] = dummy_row.values.astype("float")
    prophet_model = Prophet(yearly_seasonality = False)
    prophet_model.fit(test_row_df)
    test_row_future = prophet_model.make_future_dataframe(periods = future_periods)
    test_row_forecast = prophet_model.predict(test_row_future.iloc[-future_periods:, :])
    return(test_row_forecast.iloc[:, [0,-1]].assign(series_name = test_row_name).assign(observed_y = future_row.values).assign(y_median = time_train.iloc[row_index, -(future_periods + 30):-future_periods].median()).assign(yhat = time_train.iloc[row_index, -(future_periods + 30):-future_periods].median()))

time_train = pd.read_csv("train_2_final.csv")
time_train[:, :-62]
time_key = pd.read_csv("key_2.csv")
time_sample_sub = pd.read_csv("sample_submission_2.csv")
time_train.head()

prophet_results_full = pd.DataFrame()
for i in range(0, time_train.shape[0]):
    try:
        prophet_res = estimate_prophet(i)
        prophet_results_full = prophet_results_full.append(prophet_res, ignore_index = True)
        
    except:
        prophet_res = estimate_prophet_error(i)
        prophet_results_full = prophet_results_full.append(prophet_res, ignore_index = True)
    
prophet_results_full.to_csv("kaggle_time_train_prophet_results_all_series_train.csv", index = False)

