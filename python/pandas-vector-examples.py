import pandas as pd
import numpy as np

df = pd.DataFrame()
df["date"] = pd.DatetimeIndex(start="20160101 00:00:00", end="20160131 23:00:00", freq="T")
df["measurement"] = np.random.randn(len(df))
print(len(df))
df.head()

def naive_diff(series):
    diff_values = []
    for i in range(len(series)):
        # first value needs to be NaN
        if i == 0:
            diff_values.append(np.NaN)
        else:
            diff_values.append(series[i] - series[i-1])
    return diff_values

df["diff"] = naive_diff(df["measurement"])
df.head()

def diff_with_shift(series):
    return series - series.shift()

df["diff_2"] = diff_with_shift(df["measurement"])

df.head()

get_ipython().magic('timeit naive_diff(df["measurement"])')
get_ipython().magic('timeit diff_with_shift(df["measurement"])')

def naive_day(series):
    days = []
    for s in series:
        days.append(s.day)
    return days

get_ipython().magic('timeit naive_day(df["date"])')
get_ipython().magic('timeit df["date"].apply(lambda x: x.day)')
get_ipython().magic('timeit df["date"].dt.day')

