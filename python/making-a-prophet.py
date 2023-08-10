# load some libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib inline')

# load and clean the data
tfl = (
    pd.read_excel("tfl-daily-cycle-hires.xls", sheetname="Data")
    .rename(columns={"Day":"ds", "Number of Bicycle Hires": "y"})
    [["ds", "y"]]
    .copy()
)

tfl.ds = pd.to_datetime(tfl.ds)

tfl.head()

tfl.set_index("ds").plot(figsize=(10,4))

import fbprophet

model = fbprophet.Prophet()
model.fit(tfl);

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
model.plot(forecast);

# merge real points with forcast
real_and_forecast = pd.merge(left=forecast, right=tfl, on="ds")

# get the difference between prediction and forcast
real_and_forecast["residual"] = real_and_forecast.y - real_and_forecast.yhat

# get the range between 80% confidence intervals
real_and_forecast["uncertainty"] = real_and_forecast.yhat_upper - real_and_forecast.yhat_lower

# define an outlier as more than two intervals away from the forcast
v = 2
(
    real_and_forecast
    [real_and_forecast.residual.abs() > v * real_and_forecast.uncertainty]
    [["ds", "residual"]]
)

model.plot_components(forecast);

prepared = model.setup_dataframe(tfl)
prepared.head()

prepared.y_scaled.describe()

plt.figure(figsize=(10,4))

x = forecast.ds.values
y = forecast.trend.values
cp = model.changepoints

plt.plot(x,y)
ymin, ymax = plt.ylim()
plt.vlines(cp.values, ymin, ymax, linestyles="dashed")

model.get_changepoint_matrix()

# plot individual components
seasons = model.make_all_seasonality_features(tfl)

seasons.yearly_delim_1.plot(figsize=(10,3))
seasons.yearly_delim_2.plot(figsize=(10,3))
plt.show()

seasons.yearly_delim_5.plot(figsize=(10,3))
seasons.yearly_delim_6.plot(figsize=(10,3))
plt.show()

seasons.yearly_delim_10.plot(figsize=(10,3))
seasons.yearly_delim_11.plot(figsize=(10,3))
plt.show()

subset = [4,10]
beta = np.array(model.params["beta"])[0]

(beta[subset] * seasons[seasons.columns[subset]]).sum(axis=1).plot()

tfl.set_index("ds").ix["2015-11-01":"2016-02-15"].plot()
plt.vlines("2015-12-25", 0, 35000, linestyle="dashed")

holidays = pd.DataFrame({
    "holiday":"chirstmas",
    "ds": ["20" + str(i) + "-12-25" for i in range(11,17)],
    "lower_window":-1,
    "upper_window":2
})

holidays

# we fit as before
model_with_holidays = fbprophet.Prophet(holidays=holidays)
model_with_holidays.fit(tfl)

# and predict
forecast_with_holiday = model_with_holidays.predict(future)

# and look can examine the holiday components
model_with_holidays.plot_holidays(forecast_with_holiday);

model.params

df = model.setup_dataframe(future)

pred_trend = [model.sample_predictive_trend(df, 0) for _ in range(5)]

for i in pred_trend:
    plt.plot(i[1000:])

