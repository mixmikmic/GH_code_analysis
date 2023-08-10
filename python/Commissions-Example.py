import pandas as pd
df = pd.read_excel("https://github.com/chris1610/pbpython/blob/master/data/sample-sales-reps.xlsx?raw=true")
df.head()

df["commission"] = .02
df.head()

df.loc[df["category"] == "Shirt", ["commission"]] = .025
df.head()

df.loc[(df["category"] == "Belt") & (df["quantity"] >= 10), ["commission"]] = .04
df.head()

df["bonus"] = 0
df.loc[(df["category"] == "Shoes") & (df["ext price"] >= 1000 ), ["bonus", "commission"]] = 250, 0.045

df.ix[3:7]

df["comp"] = df["commission"] * df["ext price"] + df["bonus"]
df.head()

df.groupby(["sales rep"])["comp"].sum().round(2)

