import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

scraped_comp = pd.read_csv('../data/scraped/scraped_complaints_3_25.csv')

scraped_comp['abuse_number'] = scraped_comp['abuse_number'].apply(lambda x: x.upper())

manual = pd.read_excel('/Users/fzarkhin/OneDrive - Advance Central Services, Inc/fproj/github/database-story/scraper/manual verification.xlsx', sheetname='All manual')

manual = manual.groupby('name').sum().reset_index()

manual['name']= manual['name'].apply(lambda x: x.strip())
scraped_comp['fac_name']= scraped_comp['fac_name'].apply(lambda x: x.strip())

df = scraped_comp.groupby('fac_name').count().reset_index()[['fac_name','abuse_number']]

merge1 = manual.merge(df, how = 'left', left_on = 'name', right_on='fac_name')

merge1[merge1['count']!=merge1['abuse_number']].sort_values('abuse_number')#.sum()

manual[manual['name']=='AVAMERE AT SANDY']

scraped_comp[scraped_comp['abuse_number']=='BH116622B']

scraped_comp[scraped_comp['fac_name'].str.contains('FLAGSTONE RETIREME')]

merge2 = manual.merge(df, how = 'right', left_on = 'name', right_on='fac_name')

merge2[merge2['count']!=merge2['abuse_number']].sort_values('count')#.sum()

