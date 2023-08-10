get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

house_df = pd.read_csv("kc_house_data.csv")
house_df.head()

house_df.info()

house_df["is_basement"] = house_df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

# histogram of prices
sns.distplot(house_df["price"])

# average price for no. of bedrooms in house
grpby_bedrooms_df = house_df[["price", "bedrooms"]].groupby(by = "bedrooms", as_index = False)
grpby_bedrooms_df = grpby_bedrooms_df.mean()
grpby_bedrooms_df.head()

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.set(yscale = "log")
sns.stripplot(x = "bedrooms", y = "price", data = house_df, ax = ax1, jitter=True, palette="Blues_d")
sns.barplot(x = "bedrooms", y = "price", data = grpby_bedrooms_df, ax = ax2, palette="Blues_d")

sqft = ["sqft_living", "sqft_lot", "sqft_living15", "sqft_lot15", "sqft_above", "sqft_basement"]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize = (10, 15))
ax1.set(yscale = "log")
ax2.set(yscale = "log")
ax3.set(yscale = "log")
ax4.set(yscale = "log")
ax5.set(yscale = "log")
ax6.set(yscale = "log")

sns.regplot(x = sqft[0], y = "price", data = house_df, ax = ax1)
sns.regplot(x = sqft[1], y = "price", data = house_df, ax = ax2)
sns.regplot(x = sqft[2], y = "price", data = house_df, ax = ax3)
sns.regplot(x = sqft[3], y = "price", data = house_df, ax = ax4)
sns.regplot(x = sqft[4], y = "price", data = house_df, ax = ax5)
sns.regplot(x = sqft[5], y = "price", data = house_df, ax = ax6)

fig.tight_layout()
# sns.pairplot(house_df, y_vars = sqft, x_vars = ["price"], size = 5, kind = "reg")

grpby_is_basement_df = house_df[["is_basement", "price"]].groupby(by = "is_basement", 
                                                               as_index = False)
grpby_is_basement_df = grpby_is_basement_df.mean()

sns.barplot(x = "is_basement", y = "price", data = grpby_is_basement_df)

cols = house_df.columns.values.tolist()
cols.remove("id")
cols.remove("is_basement")
cols.remove("zipcode")

corrmat = house_df[cols].corr()

f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corrmat, square = True)

