import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/abulbasar/data/master/mobile-sales-data.csv")
df

df.info()

df.isnull()

df.isnull().sum()

df.isnull().sum(axis = 1)

df[df.isnull().sum(axis = 1) > 0]

df.dropna()

df.dropna(axis = 1)

df.fillna(df.mean())

df.fillna(df.median())

df

df["Salary"] = df.apply(lambda row: row.Salary if not np.isnan(row.Salary) else row.Age, axis = 1)
df



