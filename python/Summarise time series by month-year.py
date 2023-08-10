# import Pandas
import pandas as pd

#read time series into dataframe

df = pd.read_csv('C:data_for_examples/1.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

df.resample('MS').sum().head(12)

df.resample('AS').sum().head(12)

df.resample('AS-JUL').sum().head(12)





