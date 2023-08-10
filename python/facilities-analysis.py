import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

df = pd.read_csv('../../data/processed/facilities-3-29-scrape.csv')

df.count()[0]

df[(df['offline'].isnull())].count()[0]

df[(df['offline'].notnull())].count()[0]

df[(df['offline']>df['online']) & (df['online'].notnull())].count()[0]

df[(df['online'].isnull()) & (df['offline'].notnull())].count()[0]

df[(df['online'].notnull()) & (df['offline'].isnull())].count()[0]

df[(df['online'].notnull()) | df['offline'].notnull()].count()[0]

df[(df['offline'].isnull())].count()[0]/df.count()[0]*100

df[df['offline'].notnull()].sum()['fac_capacity']

df[df['online'].isnull()].count()[0]

over_50 = df[((df['offline']+df['online'])>50)]

over_50['total'] = over_50['online']+over_50['offline']

over_50['pct_offline'] = over_50['offline']/over_50['total']*100

over_50[over_50['facility_name']=='Avamere Health Services of Rogue Valley']

over_50.sort_values('pct_offline',ascending = False).head(10)



