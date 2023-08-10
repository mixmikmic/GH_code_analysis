import pandas as pd

df = pd.DataFrame(data=[['Red','Red','Red','Blue','Green'],
                        ['Honda','Acura','Honda','Nissan','Nissan']])
df = df.T
df.columns = ['Color','Model']
df

pd.crosstab(df['Color'],df['Model'])

