# df["Date"] > datetime(year=2015, month=4, day=1)
import pandas as pd
from datetime import datetime

sphist = pd.read_csv("../data/GP13/sphist.csv")

print(sphist["Date"].head(3))
print(sphist["Date"].dtype)

sphist["Date"] = pd.to_datetime(sphist["Date"])
print(sphist["Date"].dtype)

sphist.sort_values("Date", axis=0, ascending=True, inplace=True)

print(sphist["Date"].head(3))
print(sphist.head(3))

shifted_close = sphist["Close"].shift(periods=1, freq=None, axis=0)
#sphist["day_5"] = pd.rolling_mean(shifted_close, 5)
#sphist["day_30"] = pd.rolling_mean(shifted_close, 30)
#sphist["day_365"] = pd.rolling_mean(shifted_close, 365)
sphist["day_5"] = shifted_close.rolling(center=False,window=5).mean()
sphist["day_30"] = shifted_close.rolling(center=False,window=30).mean()
sphist["day_365"] = shifted_close.rolling(center=False,window=365).mean()

sphist["std_5"] = shifted_close.rolling(center=False,window=5).std()
sphist["std_365"] = shifted_close.rolling(center=False,window=365).std()

sphist["rday_5_365"] = sphist["day_5"] / sphist["day_365"]
sphist["rstd_5_365"] = sphist["std_5"] / sphist["std_365"]

cols = ["Date", "Close", "day_5","day_30","day_365","std_5","std_365","rday_5_365","rstd_5_365"]
ABT = sphist[cols]
ABT = ABT[ABT["Date"] > datetime(year=1951, month=1, day=2)]
ABT = ABT.dropna(axis=0)
print(ABT[ABT["Date"] > datetime(year=1951, month=1, day=2)].head())

train = ABT[ABT["Date"] < datetime(year=2013, month=1, day=1)]
test = ABT[ABT["Date"] >= datetime(year=2013, month=1, day=1)]

print(train.tail())
print(test.head())

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

features = ["day_5","day_30","day_365","std_5","std_365","rday_5_365","rstd_5_365"]
target = ["Close"]

lr = LinearRegression()
lr.fit(train[features], train[target])
predictions = lr.predict(test[features])
print(predictions[0:5])
print(test[target][0:5])

mse = mean_squared_error(test["Close"], predictions)
rmse = mse ** (1/2)
print(mse)
print(rmse)

