import pandas as pd
import re

USER_HANDLE_REGEX = re.compile(r'twitter\.com/(.+)/status/')
USER_HANDLE_REGEX.findall('http://twitter.com/malkanen/status/')

dirname="SkinDamage/SkinDamage"
df = pd.read_csv(
    "%s_processed.old.csv" % dirname
).dropna(
    subset=["GUID", "Date", "Author"]
).drop_duplicates(subset=["Date", "Author"])

df_orig = pd.read_csv("%s_noDublict.old.csv" % dirname).rename(
        columns={"Date (CST)": "Date"}
    ).assign(
        Author=lambda x: x.URL.apply(
            lambda x: "@%s" % USER_HANDLE_REGEX.findall(x)[0]
        )
    ).drop_duplicates(subset=["Date", "Author"])
df.shape, df_orig.shape

get_ipython().run_cell_magic('time', '', 'df = df.assign(date_sorted=pd.to_datetime(df.Date)).sort_values(\n    ["date_sorted", "Author"], ascending=False)\ndf_orig = df_orig.assign(date_sorted=pd.to_datetime(df_orig.Date)).sort_values(\n    ["date_sorted", "Author"], ascending=False)')

df.processedPost.head().values

df_orig.Contents.head().values

df[["GUID", "Author", "Date"]].head()

df_orig[["GUID", "Author", "Date"]].head()

df = df.drop(["date_sorted",], axis=1)
df_orig = df_orig.rename(columns={"Date": "Date (CST)"}
              ).drop(["date_sorted","Author"], axis=1)

get_ipython().run_cell_magic('time', '', 'df.drop("URL", axis=1).to_csv("%s_processed.csv" % dirname, index=False)\ndf_orig.to_csv("%s_noDublict.csv" % dirname, index=False)')

print dirname
df = pd.read_csv("%s_processed.csv" % dirname)
df_orig = pd.read_csv("%s_noDublict.csv" % dirname)
print df_orig.shape, df.shape
assert df_orig.shape[0] == df.shape[0]
df_merged = pd.concat([df, df_orig[["URL", "Contents"]]], axis=1)
print df_merged.shape
assert df_merged.shape[0] == df.shape[0]
assert (df_merged.Author != df_merged.URL.apply(lambda x: "@%s" % USER_HANDLE_REGEX.findall(x)[0])).sum() == 0

(df_merged.Author != df_merged.URL.apply(lambda x: "@%s" % USER_HANDLE_REGEX.findall(x)[0])).sum()



