import pandas as pd
df = pd.DataFrame([['Alan','Bob','Catherine'],[34, 26, 43]]).T
df.columns = ['name','age']

df

df.name = df.name.astype('category')
df.name = df.name.cat.codes
df

