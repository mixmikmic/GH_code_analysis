import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

# connect to postgres db on vb

try:
    conn = psycopg2.connect(database='vagrant', user='eruser', password = 'emergency', host='localhost', port='5432')
    print("Opened database successfully")
    
except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror)
    print(traceback.format_exc())

'''
query psql to pull number of crimes by census track by week from 2012 - 2016
inserts query result in pandas df called df2
'''

cur = conn.cursor()
QUERY='''SELECT census_tract, date_trunc('week', occ_date) AS week, count(*) AS num
    FROM crimes
    GROUP BY week, census_tract
    ORDER BY week, census_tract;'''

try:
    print("SQL QUERY = "+QUERY)
    cur.execute(QUERY)

except Exception as e:
    print(e.pgerror)
    
# Extract the column names and insert them in header
col_names = []
for elt in cur.description:
    col_names.append(elt[0])    
    
D = cur.fetchall() #convert query result to list
#pprint(D)
df = pd.DataFrame(D) #convert to pandas df
#conn.close()
df.head() # print the df head


# Create the dataframe, passing in the list of col_names extracted from the description
df2 = pd.DataFrame(D, columns=col_names)
df2.head()

#conn.close()

df2.info()

# look at number of crimes by census track
grouped_by_ct = df2.groupby('census_tract')['num'].sum().reset_index() # total crimes by census tract, resets index
grouped_by_ct.sort_values('num', ascending=False)[:25] # sort by num descending, show top 25

# barchart of top 25 census tracks with highest total crime
top_25 = grouped_by_ct.sort_values('num', ascending=False)[:25]
sns.set_style("whitegrid")
top_25_barplot = sns.barplot(x= top_25['census_tract'],y = top_25['num'], order = top_25['census_tract'], )
for item in top_25_barplot.get_xticklabels():
    item.set_rotation(90) # rotates x axis labels

# some summary stats
print('median crimes/tract = '+str(grouped_by_ct['num'].median()))
grouped_by_ct.describe()

# let's look at census tract 10600 in greater detail

df2[df2['census_tract'] == 10600.0].head()

ct_by_week_1 = df2[df2['census_tract'] == 10600.0]
#ct_by_week_1_series = pd.Series(ct_by_week_1['num'], index = ct_by_week_1['week'])
plot1 = ct_by_week_1.loc[:,['week', 'num']].plot()
plot1.set_xlabel("week")

df2[df2['census_tract'] == 5100.0].loc[:,['week', 'num']].plot()

df2[df2['census_tract'] == 8100.0].loc[:,['week', 'num']].plot()

sns.distplot(grouped_by_ct['num'])



# which census tracts have the lowest amount of crime?
grouped_by_ct.sort_values('num', ascending=False)[-75:] # sort by num descending, show top 25

low_75 = grouped_by_ct.sort_values('num', ascending=False)[-75:] 
n = sum([1 for x in low_75['num'] if x <= 3])

print('there are {} census tracts with 3 or fewer crimes'.format(n))

#let's visualize what weekly crime trends for one of those "few crime" tracts look like
df2[df2['census_tract'] == 22107.0].loc[:,['week', 'num']].plot()

df2[df2['census_tract'] == 21500.0].loc[:,['week', 'num']].plot()

# how many types of crime in db?
cur = conn.cursor()
QUERY='''SELECT DISTINCT case_desc 
    FROM crimes
    ORDER BY case_desc;'''
    
try:
    print("SQL QUERY = "+QUERY)
    cur.execute(QUERY)

except Exception as e:
    print(e.pgerror)    

# Extract the column names and insert them in header
col_names = []
for elt in cur.description:
    col_names.append(elt[0])    
    
D = cur.fetchall() #convert query result to list
#pprint(D)

# Create the dataframe, passing in the list of col_names extracted from the description
df4 = pd.DataFrame(D, columns=col_names)
print(len(df4))
df4

# try aggregating by crime type

'''
query psql to pull number of crimes by case_desc by week from 2012 - 2016
inserts query result in pandas df called df3
'''

cur = conn.cursor()
QUERY='''SELECT  case_desc, date_trunc('week', occ_date) AS week, count(*) AS num
    FROM crimes
    GROUP BY week, case_desc
    ORDER BY week, case_desc;'''


try:
    print("SQL QUERY = "+QUERY)
    cur.execute(QUERY)

except Exception as e:
    print(e.pgerror)
    
# Extract the column names and insert them in header
col_names = []
for elt in cur.description:
    col_names.append(elt[0])    
    
D = cur.fetchall() #convert query result to list
#pprint(D)

# Create the dataframe, passing in the list of col_names extracted from the description
df3 = pd.DataFrame(D, columns=col_names)
df3.head()

df3.info()

# another way to look at the crime types
for i, crime in df3.iterrows():
    print(crime['case_desc'])
    if i == 50:
        break

grouped_by_crime = df3.groupby('case_desc')['num'].sum().reset_index()     
#grouped_by_ct = df2.groupby('census_tract')['num'].sum().reset_index() # total crimes by census tract, resets index
top_25_crimes = grouped_by_crime.sort_values('num', ascending=False)[:25] # sort by num descending, show top 25  
grouped_by_crime.sort_values('num', ascending=False)[:25]

sns.set_style("whitegrid")
top_25_crimeplot= sns.barplot(y= top_25_crimes['case_desc'],x = top_25_crimes['num'], order = top_25_crimes['case_desc'])



