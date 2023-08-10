import pandas as pd
import numpy as np

load = pd.read_csv('../input/load.csv')

def normalize(x):
    return (x - x.min())/(x.min()-x.max())
def mse(actual,forecast):
    return ((actual-forecast)**2).mean()

actual,forecast = normalize(load['actual'].values),normalize(load['forecast'].values)

mse(actual=actual, forecast=forecast)

howmany_days = 30
load.iloc[:96*howmany_days].to_csv('../input/sample_load.csv', index=False)

load.loc[load['actual']==19344]



