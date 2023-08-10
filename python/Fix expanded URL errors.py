import pandas as pd

df = pd.read_csv("url_expanded.full.txt", sep="\t", header=None)
df.shape

df.head()

df.columns = ["URL", "EXPANDED", "EXPANDED_STATUS"]
df.head()

df.EXPANDED_STATUS.value_counts()

df[df.EXPANDED_STATUS == 1].head()

df[df.EXPANDED_STATUS == 3].head(100)

df[df.EXPANDED_STATUS == 3].EXPANDED.str.split("/").apply(lambda x: x[2]).value_counts()

df[(df.EXPANDED_STATUS == 3) & (df.EXPANDED.str.split("/").apply(lambda x: x[2]) == "www.huffingtonpost.com")].head()

df_err = pd.read_csv("url_expanded.error.1.txt", sep="\t", header=None)
df_err.shape

df_err.columns = ["URL", "EXPANDED", "EXPANDED_STATUS"]
df_err.head()

df_err.EXPANDED_STATUS.value_counts()

df_err[df_err.EXPANDED_STATUS == 3].EXPANDED.str.split("/").apply(lambda x: x[2]).value_counts()

df = df.set_index("URL")
df_err = df_err.set_index("URL")
df.shape, df_err.shape

df.head()

df.ix[df_err.index, ["EXPANDED", "EXPANDED_STATUS"]] = df_err[["EXPANDED", "EXPANDED_STATUS"]]

df.ix[df_err.index]["EXPANDED_STATUS"].value_counts()

df.to_csv("url_expanded.merged.txt", sep="\t")
get_ipython().system(' head url_expanded.merged.txt')



