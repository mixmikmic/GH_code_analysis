import pandas as pd
import re

df= pd.read_table('BBC.txt', header=None)

df.head()

print("Number of data points in this dataset: ", len(df))

s = df[0].str.split(' ')

for x in s:
    x[0] = re.sub('^', 'Label:', x[0])

s[0]

m = [list(w for w in x if w)for x in s]

m[0]

d = [dict(w.split(':') for w in x) for x in m]

d[0]

cols = sorted(d, key=len, reverse=True)[0].keys()
print(cols)

df = pd.DataFrame.from_records(d, index=df.index, columns=cols)

df.head()

print("Number of rows: ", len(df))

df_cols = list(df.columns)
df_cols.remove('Label')
df_cols = list(map(int, df_cols))
df_cols.sort()
df_cols = list(map(str, df_cols))
df_cols.append('Label')

df = df[df_cols]

df = df.fillna('NA')

df.head()

df.to_csv("BBC_Cleaned.csv", index=False)

