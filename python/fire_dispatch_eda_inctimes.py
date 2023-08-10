import os 
from dotenv import load_dotenv, find_dotenv
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

# walk root diretory to find and load .env file w/ AWS host, username and password
load_dotenv(find_dotenv())

# connect to postgres

try:
    conn = psycopg2.connect(database=os.environ.get("database"), user=os.environ.get("user"), 
                            password = os.environ.get("password"), 
                            host=os.environ.get("host"), port=os.environ.get("port"))
    
except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror)
    print(traceback.format_exc())


def pquery(QUERY):
    '''
    takes SQL query string, opens a cursor, and executes query in psql
    '''
    
    cur = conn.cursor()
    
    try:
        print("SQL QUERY = "+QUERY)
        cur.execute("SET statement_timeout = 0")
        cur.execute(QUERY)
        # Extract the column names and insert them in header
        col_names = []
        for elt in cur.description:
            col_names.append(elt[0])    
    
        D = cur.fetchall() #convert query result to list
        #pprint(D)
        #conn.close()
        # Create the dataframe, passing in the list of col_names extracted from the description
        return pd.DataFrame(D, columns=col_names)

    except Exception as e:
        print(e.pgerror)
    
   

# look at inctimes table first
QUERY1='''
SELECT inctimes_id, timedesc_id, incident_id, realtime,
 EXTRACT(DOW FROM realtime) AS dow,
 EXTRACT(MONTH FROM realtime) AS month,
 EXTRACT(YEAR FROM realtime) AS year
FROM inctimes;
'''

df1 = pquery(QUERY1)

df1.info()

df1.head(15)

# Let's audit the data for duplicates where inctimes_id and timedesc_id are the same. 
# This code shows all rows that have a duplicate pair of incident_id, timedesc_id

df1.duplicated(['incident_id','timedesc_id'],keep = False).head(25)

# print only rows where a duplicate incident_id/timedesc_id pair exists
df1[df1.duplicated(['incident_id','timedesc_id'],keep = False)]

# this code does that (showing only first 25 rows)
df1[df1.duplicated(['incident_id','timedesc_id'],keep = 'first') == False].head(25)

df1_nd = df1[df1.duplicated(['incident_id','timedesc_id'],keep = 'first') == False]

# after eliminating duplicates, what does the distribution of timedesc_ids look like?
df1_nd.groupby('timedesc_id')['timedesc_id'].count()

# after eliminating duplicates, what does the distribution of timedesc_ids look like?
pd.crosstab(index='count', columns=df1_nd['timedesc_id'])

# same crosstab data as a %
tab_norm = pd.crosstab(index='count', columns=df1_nd['timedesc_id']).apply(lambda r: r/r.sum(), axis=1)
tab_norm

tab_norm.plot.barh(stacked = True)

# this function calculates time deltas for each incident id in the df
# iterating over the groupby object was very helpful

def time_delta(dataframe):
    '''caluclates the time delta between two different timedesc_ids for all groups'''
    count = 0
    group_dict = {}
    for name, group in dataframe.groupby('incident_id'):
        delta_dict = {}
        #print(name)
        #print(group)
        df_t = group.set_index('timedesc_id') # reset the group df to index on timedesc_id

        try:
            delta_dict['delta3-5'] = df_t.loc[5,'realtime']-df_t.loc[3,'realtime']
            delta_dict['dow'] = df_t.loc[3,'dow']
            delta_dict['month'] = df_t.loc[3,'month']
            
        except KeyError:
            delta_dict['delta3-5'] = None
            
        try:
            delta_dict['delta3-10'] =  df_t.loc[10,'realtime']-df_t.loc[3,'realtime']
            
        except KeyError:
            delta_dict['delta3-10'] = None   
        
        group_dict[name] = delta_dict        

    return group_dict
     

# use DataFrame.from_dict method to convert the dictionary object returned in the time_delta function to 
# a pandas df. Note using argument: orient = 'index' makes the dictionary keys into column headers.

#print(time_delta(df1_nd))
time_delta_df = pd.DataFrame.from_dict(time_delta(df1_nd), orient = 'index')
time_delta_df.index.name = 'incident_id'
time_delta_df.head(100)

time_delta_df.describe()

'''
I noticed there are some negative timedelta values which are obviously impossible and should be thrown out.
In addition, pandas cannot plot histgram by just using: time_delta_df.plot()
So we need to do some timedelta conversion to a more easily plotted numerical type.
My approach was to convert the deltas to decimal expression of minutes. This link was helpful:
http://www.datasciencebytes.com/bytes/2015/05/16/pandas-timedelta-histograms-unit-conversion-and-overflow-danger/
'''
print(time_delta_df['delta3-5'].dropna() / pd.Timedelta(minutes=1)) # note that I dropped missing values from the time_delta_df

# remove null and negative values
# this link was helpful: http://stackoverflow.com/questions/24214941/python-pandas-dataframe-filter-negative-values
clean3_5 = time_delta_df[time_delta_df.loc[:,'delta3-5'] > 
                         pd.Timedelta(0)].loc[:,'delta3-5'].dropna()/pd.Timedelta(minutes=1)
clean3_10 = time_delta_df[time_delta_df.loc[:,'delta3-10'] > 
                         pd.Timedelta(0)].loc[:,'delta3-10'].dropna()/pd.Timedelta(minutes=1)

clean3_5.describe()

clean3_5.hist(bins=range(0,30,1))
plt.xlabel('Minutes elapsed from timedesc_id 3 to 5 (Dispatched to On-scene)')
plt.ylabel('Count')

clean3_10.hist(bins=range(0,250,5))
plt.xlabel('Minutes elapsed from timedesc_id 3 to 10 (Dispatched to Arrive Destination)')
plt.ylabel('Count')

clean3_10.describe()

time_delta_df.head(10)

# clean time_delta_df of negative times values and Nulls
time_delta_clean310 = time_delta_df[time_delta_df.loc[:,'delta3-10'] > 
                         pd.Timedelta(0)]

time_delta_clean310.head(10)

# summary stats for 3-10 response time of each day of week
time_delta_clean310.groupby('dow')['dow', 'delta3-10'].describe()

# try mapping numbers to strings for days of the week
# this was helpful: http://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict
dow_map = {0.0:'sunday',
           1.0:'monday', 
           2.0:'tuesday', 
           3.0:'wednesday', 
           4.0:'thursday', 
           5.0:'friday', 
           6.0:'saturday'}

td310_mapped = time_delta_clean310.replace({'dow':dow_map}).groupby('dow')['dow', 'delta3-10']

# boxplot of 3-10 response time for each day of week
sns.set_style("whitegrid")
plt.figure(figsize=(14,6))
bplot = sns.boxplot(x='dow', y='delta3-10', data=time_delta_clean310.replace({'dow':dow_map}), whis=[5,95])
title = 'Distribution of 3-10 Reponse Times For Each Day of the Week'
bplot.set_title(title, fontsize=15)
bplot.set_xlabel('Day of Week', fontsize=12)
bplot.set_ylabel('Response Time (3-10)', fontsize=12)
bplot.tick_params(axis='both', labelsize=8)
sns.despine(left=True) 

# summary stats of 3-10 response time for each month of the year
month_map = {1.0:'jan',2.0:'feb',3.0:'mar',4.0:'apr',5.0:'may',6.0:'jun',
             7.0:'jul',8.0:'aug',9.0:'sep', 10.0:'oct', 11.0:'nov', 12.0:'dec'}

time_delta_df.replace({'month':month_map}).groupby('month')['month', 'delta3-10'].describe()

# boxplot of 3-10 response time by month
# decided not to map by month since it affects order of months in boxplot
sns.set_style("whitegrid")
plt.figure(figsize=(14,6))
gb_month310 = time_delta_df.groupby('month')['month', 'delta3-10']
bplot = sns.boxplot(x='month', y='delta3-10', data=time_delta_df, whis=[5,95])
title = 'Distribution of 3-10 Reponse Times For Each Month of the Year'
bplot.set_title(title, fontsize=15)
bplot.set_xlabel('Month', fontsize=12)
bplot.set_ylabel('Response Time (3-10)', fontsize=12)
bplot.tick_params(axis='both', labelsize=8)
sns.despine(left=True) 

# show descriptions of dispatch events as a quick reference
QUERY2='''SELECT * FROM timedesc;
  '''

incident_types = pquery(QUERY2)

incident_types

conn.close()

