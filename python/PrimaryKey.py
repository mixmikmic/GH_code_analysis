import pandas as pd
a = pd.read_csv('sample data.csv')
a.head()

import primarykey as pk 
pk.primarykey(a)

import preparation as pr
pr.timestamp(a,'Income')

