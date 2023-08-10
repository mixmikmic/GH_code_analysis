import pandas as pd
import copy
import matplotlib.pyplot as plt 
import seaborn as sns
from seaborn import color_palette, set_style, palplot
#plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

search_result_path ='./data/search_results.csv'
df_search = pd.read_csv(search_result_path)
df_search.drop("Unnamed: 0",axis=1,inplace= True)
df_search.doc_id = df_search.doc_id.astype(str)
df_search.head()

df_search.fillna(0,inplace=True)   ## replac NaN with 0 
search_keys = copy.deepcopy(df_search.columns.values.tolist())[4:]      ## get all keywords 
# here you can do some filter, for instance get ride of some keys that are revelent 
search_keys.remove('restructuring')  ## delete restructuring, it is everywhere, not very useful
df_search['total_key_frequency'] = df_search[search_keys].sum(axis=1)   ## generage a total keyword sum variable 

df_agg = df_search.groupby(['doc_id'],as_index = False)['total_key_frequency'].sum()
#df_agg = df_agg[df_agg.total_key_frequency != 0 ]
df_agg.head()

#### Merge metadata filds 
meta_path ='./data/Staff_reports_meta_all.xlsx'
df_meta = pd.read_excel(meta_path,'Sheet1')
df_meta.head()

df = pd.merge(df_agg,df_meta,on='doc_id')  ## merge, inner join
print("Total Number of Documents: {}".format(len(df)))
df.head()

df_agg = df.groupby(['year_final','imf_country_name','country_code','income'],as_index = False)['total_key_frequency','document_word_count'].sum()
#df_agg.index = pd.to_datetime(df_agg['year_final'],format='%Y')
df_agg['freq_norm'] = df_agg['total_key_frequency']/df_agg['document_word_count']*1000    ## keyword frequencey per 1000 words 
df_agg.head(10)

countryname= "United States"
countrycode = 111

df_plot = df_agg[df_agg['country_code']==countrycode].copy()
df_plot.plot(x='year_final',y='freq_norm',figsize=(13,6),title=countryname + " keywords frequency (per 1000 words)")

