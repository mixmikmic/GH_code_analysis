get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np

wiki_tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_Internet_connection_speeds')

type(wiki_tables)

type(wiki_tables[0])

wiki_tables[0].head()

table_final = wiki_tables[0][2:]
table_final.columns = ['rank','region','value']
table_final['region'] = table_final['region'].str.lower()
table_final.replace('united states', 'united states of america', inplace=True)
table_final['value'] = table_final['value'].astype(float)
table_final

table_final.dtypes

### load the rpy2 extension
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

get_ipython().run_line_magic('R', '-i table_final')

get_ipython().run_cell_magic('R', "-w 800 -h 600 -u px # instead of px, you can also choose 'in', 'cm', or 'mm'", 'df <- as.data.frame(table_final)\nlibrary(choroplethr)\n\ncountry_choropleth(table_final)')

