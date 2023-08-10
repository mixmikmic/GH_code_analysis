import sys

print(sys.version)

import pandas as pd

path = r"https://raw.githubusercontent.com/abulbasar/data/master/insurance.csv"

df = pd.read_csv(path)

df.head()

type(df)

df.info()

df.shape

df.shape[0]

type(df.dtypes)

df.dtypes

df.columns

df.tail()

df[df.region == "northwest"].sample(10)

df.query("region == 'northwest'").sample(10)

df.iloc[5:10, :]

df.iloc[5:10, 2:4]

df.loc[5:10, ["charges", "region"]]

df[["charges", "region"]]

df.sort_values("charges", ascending=False)

df.groupby("region")["charges"].mean()

import numpy as np

df.groupby("region")["charges"].agg([np.mean, len])

df["region2"] = df["region"].apply(lambda s: str.upper(s))
df.head()

df["charge_by_age"] = df.apply(lambda row: row.charges / row.age, axis = 1)
df.head()

16884.92400 / 19

get_ipython().magic('pinfo pd.merge')

movies = pd.read_csv("/data/ml-latest-small/movies.csv")
movies.info()

movies.head()

ratings = pd.read_csv("/data/ml-latest-small/ratings.csv")
ratings.info()

ratings.head()

rating_agg = ratings.groupby("movieId").rating.agg([np.mean, len]).reset_index().query("len>=100")
rating_agg.head()

pd.merge(movies, rating_agg, on = "movieId")[["movieId", "title", "mean"]].sort_values("mean", ascending = False).iloc[:10, :]



