import pandas as pd

snl = pd.read_csv('database/snldb/db/snl_title.csv')
snl.head()

clean_idx = snl['title'].notnull()

snl[clean_idx] 

sk34 = snl.set_index(['sid', 'eid'])

sk34.head()

sk34.sort_index(inplace=True)

sk34.head()

sk34.loc[[3, 4]]

sk34.loc[3:5]

