from censusreporter_api import * # a few convenience methods elsewhere in this repo. Not a full pypi project at the moment

df = get_dataframe(tables='B01001',geoids='040|01000US',column_names=True,level=1)
df.sort('Total', ascending=False).head(5)

df = get_dataframe(tables='B01001',geoids='040|02000US2',column_names=True,level=1)
df['pct_female'] = df['Female'] / df['Total']
df.sort('pct_female',ascending=False).head(5)

df = get_dataframe(tables='B01001',geoids='050|04000US47',column_names=True,level=1)
df['pct_male'] = df['Male'] / df['Total']
df.sort('pct_male',ascending=False).head(5)

