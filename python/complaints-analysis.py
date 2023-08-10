import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_colwidth', -1)

df = pd.read_csv('../../data/processed/complaints-3-29-scrape.csv')

df.count()[0]

df[df['public']=='offline'].count()[0]

df[df['public']=='online'].count()[0]

df[df['public']=='offline'].count()[0]/df.count()[0]*100

df[(df['outcome']=='Exposed to Potential Harm') | (df['outcome']=='No Negative Outcome')].count()[0]

df[(df['outcome']=='Exposed to Potential Harm') |
   (df['outcome']=='No Negative Outcome')].count()[0]/df[df['public']=='offline'].count()[0]*100

totals = df.groupby(['omg_outcome','public']).count()['abuse_number'].unstack().reset_index()

totals.fillna(0, inplace = True)

totals['total'] = totals['online']+totals['offline']

totals['pct_offline'] = round(totals['offline']/totals['total']*100)

totals.sort_values('pct_offline',ascending=False)

df['outcome_notes'].fillna('', inplace = True)

df[(df['outcome_notes'].str.contains('constitute neglect|constitutes neglect|constitute abuse|constitutes abuse|constitutes exploitation|constitutes financial exploitation')) & (df['public']=='offline')].count()[0]

df[(df['omg_outcome']=='Potential harm') & (df['fine']>0) & (df['public']=='offline')].count()[0]

