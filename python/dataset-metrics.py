import pandas as pd
import configparser
import requests as req
pd.set_option('display.max_colwidth',140)
pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

# bring in the information about all resources on data.austintexas.gov as of 3 Oct 2016; 
# file source = socrata resource for admins called "Dataset of Datasets", accessed 3 Oct 2016
# create a dataframe
df = pd.read_csv('all_views_20161003.csv')
df.head(3)

# check on the number of records in the file
print("The number of Socrata resources in this file is \n{}.".format(df.count()))

# take a look at the values available in the Type column
df['Type'].unique()

# an example of what's in a record
df.columns.tolist()

df.info()

#so we can drop ContactEmail (no data)
df.drop(labels=['ContactEmail'], inplace=True, axis=1)
df.columns

# we want to filter out the resources that aren't tables. And we don't want derived views. 

only_tables = df[(df.Type == 'table') &
   (df['Derived View']==False) &
    (df.Domain=='data.austintexas.gov')].copy(deep=True)
only_tables

# let's find out how many tables there are
len(only_tables)

#add a column for urls
urls = []
for x in only_tables['U ID']:
    urls.append('https://data.austintexas.gov/api/views/{}/metrics.json?start=1451606400000&end=1475539199999'.format(x))
only_tables['metrics_urls']=urls
only_tables.head()

# Simple plot of Visits + Downloads.  This doesn't reveal complexity, necessity or structure.
pd.Series(only_tables.Visits+only_tables.Downloads).sort_values().plot(kind='bar')

# get ready to call the Socrata API. don't store password in the notebook
config = configparser.ConfigParser()
config.read('secrets.txt')
user = config['socrata']['u']
password = config['socrata']['p']

# make the call for each url in the list. store each response in a dictionary
table_metrics_ytd = []
for u in only_tables.metrics_urls:
    t = {}
    r = req.get(u, auth=(user, password))
    i = u[39:48]
    l = 'https://data.austintexas.gov/d/' + i
    f = {'fetched_url': u, 'id': i, 'dataset_page_url': l}
    d = r.json()
    f.update(d)
    table_metrics_ytd.append(f)
print('made ' + str(len(table_metrics_ytd)) + ' api calls.')

# load the data we just got into a data frame and check it out
df2 = pd.DataFrame(table_metrics_ytd)
df2.columns

# check to see if any url calls returned an error

df2['code'].unique()

# rut-roh!

df2['message'].unique()

# how many datasets returned an error?
df2[['code', 'id']][df2['code'] == 'not_found'].groupby('code').count()

# make a list of them

df2[['id']][df2['code'] == 'not_found'].values

# it looks like our admin "dataset of datasets" includes resources that we federate from other places

df['Domain'].unique()

# let's see if the number of state datasets matches the number of errors we encountered

a = df[['U ID', 'Type', 'Domain']][df['Type'] == 'table'][df['Domain'] == 'data.texas.gov']
state_table_ids = a['U ID'].values
print(len(state_table_ids)) # 79 makes sense

# now let's drop those results out of our dataframe

atx_table_metrics_ytd = []
for i in table_metrics_ytd:
    if i['id'] not in state_table_ids:
        atx_table_metrics_ytd.append(i)
len(atx_table_metrics_ytd) + len(state_table_ids) == len(table_metrics_ytd) # it all adds up! hooray

# write the city metrics to a csv so more people can explore them

x = atx_table_metrics_ytd
keys = df2.columns
with open('table_metrics_ytd.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(x)



