import os 
from dotenv import load_dotenv, find_dotenv
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ggplot # testing out some ggplot goodness in python
get_ipython().magic('matplotlib inline')

# walk root diretory to find and load .env file w/ AWS host, username and password
load_dotenv(find_dotenv())

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
        return pd.DataFrame(D, columns=col_names)

    except Exception as e:
        print(e.pgerror)
            
    finally:
        conn.close()

# get all rows from census block to fma lookup table 
QUERY1='''SELECT *
FROM fmac_proportion;
'''

df1 = pquery(QUERY1)

df1.info()

df1.head(25)

# try joining census total population by census block group to fma
QUERY2='''SELECT f.fma,
  round(sum(c.estimate_total*f.overlap_cbg)) AS fma_population_total
FROM fmac_proportion f INNER JOIN census_total_population c
ON f.c_block = c.id2
GROUP BY f.fma
ORDER BY fma_population_total DESC
'''

df2 = pquery(QUERY2)

df2.info()

df2

# try joining census housing_tenure by census block group to fma
QUERY3='''SELECT f.fma,
  round(sum(c.estimate_total_households*f.overlap_cbg)) AS total_households,
  round(sum(c.estimate_total_owner_occupied*f.overlap_cbg)) AS total_owner_occupied,
  round(sum(c.estimate_total_renter_occupied*f.overlap_cbg)) AS total_renter_occupied
FROM fmac_proportion f INNER JOIN census_housing_tenure c
ON f.c_block = c.id2
GROUP BY f.fma
ORDER BY total_households DESC
'''

df3 = pquery(QUERY3)

df3.info()

df3

# try joining census estimate_median_household_income by census block group to fma
# note this is not a statistically valid query since we can't simply apply a weighted average to medians
# but it's at least one approach to approximate a median for fmas

QUERY4='''SELECT f.fma,
  round(sum(c.estimate_median_household_income*f.overlap_fma)) AS median_household_income
FROM fmac_proportion f INNER JOIN census_median_household_income c
ON f.c_block = c.id2
GROUP BY f.fma
ORDER BY median_household_income DESC
'''

df4 = pquery(QUERY4)

df4.info()

df4

# try joining census_household_income by census block group to fma to look at income distribution

QUERY5='''SELECT f.fma,
  round(sum(c.total_less_than_10000*f.overlap_fma)) AS less_than_10000,
  round(sum(c.total_10000_to_14999*f.overlap_fma)) AS fr_10000_to_14999,
  round(sum(c.total_15000_to_19999*f.overlap_fma)) AS fr_15000_to_19999,
  round(sum(c.total_20000_to_24999*f.overlap_fma)) AS fr_20000_to_24999,
  round(sum(c.total_25000_to_29999*f.overlap_fma)) AS fr_25000_to_29999,
  round(sum(c.total_30000_to_34999*f.overlap_fma)) AS fr_30000_to_34999,
  round(sum(c.total_35000_to_39999*f.overlap_fma)) AS fr_35000_to_39999,
  round(sum(c.total_40000_to_44999*f.overlap_fma)) AS fr_40000_to_44999,
  round(sum(c.total_45000_to_49999*f.overlap_fma)) AS fr_45000_to_49999,
  round(sum(c.total_50000_to_59999*f.overlap_fma)) AS fr_50000_to_59999,
  round(sum(c.total_60000_to_74999*f.overlap_fma)) AS fr_60000_to_74999,
  round(sum(c.total_75000_to_99999*f.overlap_fma)) AS fr_75000_to_99999,
  round(sum(c.total_100000_to_124999*f.overlap_fma)) AS fr_100000_to_124999,
  round(sum(c.total_125000_to_149999*f.overlap_fma)) AS fr_125000_to_149999,
  round(sum(c.total_150000_to_199999*f.overlap_fma)) AS fr_150000_to_199999,
  round(sum(c.total_200000_or_more*f.overlap_fma)) AS fr_200000_or_more
FROM fmac_proportion f INNER JOIN census_household_income c
ON f.c_block = c.id2
GROUP BY f.fma
ORDER BY f.fma
'''

df5 = pquery(QUERY5)

df5.info()

df5

for i in df5['fma']:
    df5.loc[df5['fma'] == i,'less_than_10000':].plot(kind = 'bar',title = "fma#= "+str(i))
    # Shrink current axis by 20%
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
            



