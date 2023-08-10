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

# ucomment to check environment variables
#%env

# connect to postgres
def pgconnect():
    try:
        conn = psycopg2.connect(database=os.environ.get("database"), user=os.environ.get("user"), 
                            password = os.environ.get("password"), 
                            host=os.environ.get("host"), port=os.environ.get("port"))
        print("Opened database successfully")
        return conn
    
    except psycopg2.Error as e:
        print("I am unable to connect to the database")
        print(e)
        print(e.pgcode)
        print(e.pgerror)
        print(traceback.format_exc())
        return None

def pquery(QUERY):
    '''
    takes SQL query string, opens a cursor, and executes query in psql
    '''
    conn = pgconnect()
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
        # Create the dataframe, passing in the list of col_names extracted from the description
        conn.close()
        return pd.DataFrame(D, columns=col_names)

    except Exception as e:
        print(e.pgerror)
        conn.close()

QUERY1='''SELECT incident.incident_id, incsitfoundclass.incsitfoundclass_id, incsitfoundclass.description
FROM incident
  INNER JOIN incsitfound
    ON incident.incsitfoundprm_id = incsitfound.incsitfound_id
  LEFT JOIN incsitfoundsub
    ON incsitfound.incsitfoundsub_id = incsitfoundsub.incsitfoundsub_id
  LEFT JOIN incsitfoundclass
    ON incsitfoundsub.incsitfoundclass_id = incsitfoundclass.incsitfoundclass_id;
'''

df1 = pquery(QUERY1)

df1.info()

df1.head()

df1.groupby('description')['description'].count().sort_values(ascending = 0)

sns.set_style("whitegrid")
sns.countplot(y='description', data = df1)

# incident by desciption as % of total
tab = pd.crosstab(index='count', columns = df1['description']).apply(lambda r: r/r.sum(), axis=1)
tab

tab.plot.bar(stacked = True)

QUERY2='''
SELECT incident.typenaturecode_id, typenaturecode.description, 
  count(incident.typenaturecode_id) as num
    FROM typenaturecode LEFT JOIN incident
    ON incident.typenaturecode_id = typenaturecode.typenaturecode_id
  GROUP BY incident.typenaturecode_id,typenaturecode.description
  ORDER BY num DESC;
'''

df2 = pquery(QUERY2)

df2.info()

df2[:50]

sns.set_style("white") 
brplot =  sns.barplot(x="num", y="description", data=df2[:25])
title = ('Incidents sorted by Top 25 typenaturecode')
# Add title with space below for x-axix ticks and label
brplot.set_title(title, fontsize=15, y=1.06)
brplot.set_ylabel('Nature Code Type', fontsize=12, rotation=90) # rota
brplot.set_xlabel('Count', fontsize=12)
brplot.tick_params(axis='both', labelsize=10)



