import os 
from dotenv import load_dotenv, find_dotenv
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import geoplot as gplt
import geopandas as gpd

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
    takes SQL query string, opens a cursor, executes query in psql, and pulls results into pandas df
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

'''
This function takes an SQL query, connects to postgres, 
and creates a geodataframe containing the spatial data
You use it like the pquery() function above, 
but call it when you want to incorporate a column with 
shape data into a dataframe
'''

from geopandas import GeoSeries, GeoDataFrame

def gpd_query(QUERY):
    '''
    takes SQL query string, connects to postgres, and creates geopandas dataframe
    '''
    conn = pgconnect()
    cur = conn.cursor()
    
    try:
        print("SQL QUERY = "+QUERY+'\r\n')
        geo_df = GeoDataFrame.from_postgis(QUERY, 
        conn, geom_col='geom', crs={'init': u'epsg:4326'}, 
        coerce_float=False)
        
        print("created geopandas dataframe!")
        return geo_df
        
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

df1.head(10)

# try joining census total population by census block group to fma
# include fma geometry in query

QUERY2='''SELECT f.fma,
  CAST(round(sum(c.estimate_total*f.overlap_cbg)) AS INTEGER) AS fma_population_total, s.geom
FROM fmac_proportion f 
INNER JOIN census_total_population c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY fma_population_total DESC
'''

df2 = gpd_query(QUERY2) # note the use of gdp_query() since we want to return a geopandas dataframe

df2.info()

df2

# map of population by fma
import geoplot.crs as gcrs # allow different projections

gplt.choropleth(df2,
                hue=df2['fma_population_total'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='Purples',
                linewidth=0.5,
                k=9,
                legend=True,
                scheme='equal_interval',
                figsize=(10, 10))

plt.title("Total population by FMA")

# explore household tenure (owner-occupied vs renter occupied) by FMA

QUERY3='''SELECT f.fma,
  CAST(round(sum(c.estimate_total_households*f.overlap_cbg)) AS INTEGER) AS total_households,
  CAST(round(sum(c.estimate_total_owner_occupied*f.overlap_cbg)) AS INTEGER) AS total_owner_occupied,
  (sum(c.estimate_total_owner_occupied*f.overlap_cbg))/(sum(c.estimate_total_households*f.overlap_cbg))
  AS percent_owner_occupied,
  CAST(round(sum(c.estimate_total_renter_occupied*f.overlap_cbg)) AS INTEGER) AS total_renter_occupied,
  (sum(c.estimate_total_renter_occupied*f.overlap_cbg)/sum(c.estimate_total_households*f.overlap_cbg))
    AS percent_renter_occupied,
  s.geom
FROM fmac_proportion f INNER JOIN census_housing_tenure c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY total_households DESC
'''

df3 = gpd_query(QUERY3)  

df3.info()

df3[df3.columns[:5]] # not printing out geom column which is in last column

gplt.choropleth(df3,
                hue=df3['percent_owner_occupied'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='Greens',
                linewidth=0.5,
                #k=9,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of households that are owner-occupied by FMA")

# just for visual comparison, here is the same choropleth map 
# of FMA by percent owner occupied using matplotlib
# NOTE: I had to remove the legend to prevent matplotlib from 
# throwing an error from scaling based on a float number

df3.plot(column='percent_owner_occupied', cmap= 'RdPu',figsize=(10,10))

# try joining census estimate_median_household_income by census block group to fma
# note this is not a statistically valid query since we can't simply apply a weighted average to medians
# but it's at least one approach to approximate a median for fmas

QUERY4='''SELECT f.fma,
  round(sum(c.estimate_median_household_income*f.overlap_fma)) AS median_household_income,
  s.geom
FROM fmac_proportion f 
  INNER JOIN census_median_household_income c
  ON f.c_block = c.id2
  INNER JOIN fma_shapes s
  ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY median_household_income DESC
'''

df4 = gpd_query(QUERY4)

#df4.info()

df4[df4.columns[:2]]

gplt.choropleth(df4,
                hue=df4['median_household_income'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='RdBu',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Estimate of median income by FMA")

# explore health insurance status by FMA
QUERY5='''SELECT f.fma,
  round(sum(c.with_health_insurance*f.overlap_cbg)) AS with_health_insurance,
  ((sum(c.with_health_insurance*f.overlap_cbg))/(sum(c.with_health_insurance*f.overlap_cbg)
      +(sum(c.no_health_insurance*f.overlap_cbg)))) AS percent_with_health_insurance,
  round(sum(c.no_health_insurance*f.overlap_cbg)) AS no_health_insurance,
  ((sum(c.no_health_insurance*f.overlap_cbg))/(sum(c.with_health_insurance*f.overlap_cbg)
      +(sum(c.no_health_insurance*f.overlap_cbg)))) AS percent_no_health_insurance,
  s.geom    
FROM fmac_proportion f INNER JOIN census_health_insurance c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_no_health_insurance DESC
'''

df5 = gpd_query(QUERY5)

df5[df5.columns[:5]]

gplt.choropleth(df5,
                hue=df5['percent_no_health_insurance'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='OrRd',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of individuals without health insurance by FMA")

# explore educational attainment by FMA
# aggregate by calculating % of individuals with just college degree, % with college degree or higher

QUERY6= '''SELECT f.fma,
  round(sum(c.total*f.overlap_cbg)) AS total,
  round(sum(c.bachelor_degree*f.overlap_cbg)) AS college_grad,
  ((sum(c.bachelor_degree*f.overlap_cbg))/(sum(c.total*f.overlap_cbg)
      )) AS percent_college_grad,
  (((sum(c.bachelor_degree*f.overlap_cbg))
  +(sum(c.master_degree*f.overlap_cbg))
  +(sum(c.professional_school_degree*f.overlap_cbg))
  +(sum(c.doctorate_degree*f.overlap_cbg)))) AS college_grad_or_higher,   
  (((sum(c.bachelor_degree*f.overlap_cbg))
  +(sum(c.master_degree*f.overlap_cbg))
  +(sum(c.professional_school_degree*f.overlap_cbg))
  +(sum(c.doctorate_degree*f.overlap_cbg)))
  /(sum(c.total*f.overlap_cbg))) AS percent_college_grad_or_higher,
  s.geom   
FROM fmac_proportion f INNER JOIN census_educational_attainment c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_college_grad_or_higher DESC
'''


df6 = gpd_query(QUERY6)

df6[df6.columns[0:6]] 

gplt.choropleth(df6,
                hue=df6['percent_college_grad_or_higher'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='YlGnBu',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of adults 25 and older with a college degree or higher by FMA")

# explore food stamps by FMA
QUERY7 = '''SELECT f.fma,
  round(sum(c.total*f.overlap_cbg)) AS total,
  round(sum(c.hh_rec_fs*f.overlap_cbg)) AS rec_food_stamps,
  ((sum(c.hh_rec_fs*f.overlap_cbg))/(sum(c.total*f.overlap_cbg)
      )) AS percent_rec_fs,
  round(sum(c.hh_dn_rec_fs*f.overlap_cbg)) AS dn_red_food_stamps,
  ((sum(c.hh_dn_rec_fs*f.overlap_cbg))/(sum(c.total*f.overlap_cbg)
      )) AS percent_dn_rec_fs,
  s.geom   
FROM fmac_proportion f INNER JOIN census_food_stamps c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY rec_food_stamps DESC
'''

df7 = gpd_query(QUERY7)

df7[df7.columns[:5]]

gplt.choropleth(df7,
                hue=df7['percent_rec_fs'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='OrRd',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of households receiving food stamps in last 12 months by FMA")

# explore household language by FMA
# I chose to aggregate by housholds that speak English only
# and also limited english speaking status households (lesh)

QUERY8='''SELECT f.fma,
  round(sum(c.estimate_total*f.overlap_cbg)) AS total,
  round(sum(c.english_only*f.overlap_cbg)) AS english_only,
  ((sum(c.english_only*f.overlap_cbg))/(sum(c.estimate_total*f.overlap_cbg))) AS percent_english_only,
  (round((sum(c.spanish_lesh*f.overlap_cbg)) + (sum(c.Other_Indo_Euro_lesh*f.overlap_cbg)) + 
  (sum(c.Asian_Pacific_Island_lesh*f.overlap_cbg)) + (sum(c.other_lesh*f.overlap_cbg)))) AS total_lesh,
  (((sum(c.spanish_lesh*f.overlap_cbg)) + (sum(c.Other_Indo_Euro_lesh*f.overlap_cbg)) + 
  (sum(c.Asian_Pacific_Island_lesh*f.overlap_cbg)) + (sum(c.other_lesh*f.overlap_cbg)))/
  (sum(c.estimate_total*f.overlap_cbg))) AS percent_total_lesh,
  s.geom    
FROM fmac_proportion f INNER JOIN census_household_language c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_total_lesh DESC
'''


df8 = gpd_query(QUERY8)

df8[df8.columns[:6]]

gplt.choropleth(df8,
                hue=df8['percent_total_lesh'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='GnBu',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of households that have limited english speaking status")

# explore race by FMA 
# I chose to summarize by all race groups in the db table, but show on map as fraction of non-white

QUERY9='''SELECT f.fma, 
  round(sum(c.total*f.overlap_cbg)) AS total,
  round(sum(c.White_alone*f.overlap_cbg)) AS white_alone,
  round(sum(c.Black_alone*f.overlap_cbg)) AS black_alone,
  round(sum(c.American_Indian_Alaska_Native_alone*f.overlap_cbg)) AS natam_alkn_alone,
  round(sum(c.Asian_alone*f.overlap_cbg)) AS asian_alone,
  round(sum(c.Native_Hawaiian_Pacific_Islander_alone*f.overlap_cbg)) AS haw_pacis_alone,
  round(sum(c.Some_other_alone*f.overlap_cbg)) AS other_alone,
  round(sum(c.Two_or_more_races*f.overlap_cbg)) AS two_or_more,
  (((sum(c.total*f.overlap_cbg)) - (sum(c.White_alone*f.overlap_cbg)))/((sum(c.total*f.overlap_cbg)))) 
      AS percent_non_white,
  s.geom   
FROM fmac_proportion f INNER JOIN census_race c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_non_white DESC
'''

df9 = gpd_query(QUERY9)

df9[df9.columns[:-1]]

gplt.choropleth(df9,
                hue=df9['percent_non_white'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='OrRd',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of individuals that are non-white by FMA")

# explore poverty status by FMA

QUERY10='''SELECT f.fma, 
  round(sum(c.estimate_total*f.overlap_cbg)) AS total,
  round(sum(c.total_below*f.overlap_cbg)) AS below_pov_line,
  ((sum(c.total_below*f.overlap_cbg))/(sum(c.estimate_total*f.overlap_cbg))) AS percent_below_pov,
  round(sum(c.total_above*f.overlap_cbg)) AS above_pov_line,
  ((sum(c.total_above*f.overlap_cbg))/(sum(c.estimate_total*f.overlap_cbg))) AS percent_above_pov,
  s.geom   
FROM fmac_proportion f INNER JOIN census_poverty_status_individuals c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_below_pov DESC
'''

df10 = gpd_query(QUERY10)

df10[df10.columns[:-1]]

gplt.choropleth(df10,
                hue=df10['percent_below_pov'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='YlOrRd',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of households that are below poverty level BY FMA")

# explore households with member 65+ years by FMA

QUERY11='''SELECT f.fma,
  round(sum(c.totals*f.overlap_cbg)) AS total,
  round(sum(c.oneplus_people_65plus*f.overlap_cbg)) AS member_65plus,
  ((sum(c.oneplus_people_65plus*f.overlap_cbg))/(sum(c.totals*f.overlap_cbg)))
       AS percent_member_65plus,   
  s.geom   
FROM fmac_proportion f INNER JOIN census_households_65plus c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_member_65plus DESC
'''

df11 = gpd_query(QUERY11)

df11[df11.columns[:-1]]

gplt.choropleth(df11,
                hue=df11['percent_member_65plus'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='YlGnBu',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))

plt.title("Fraction of households with at least one member aged 65+ by FMA")

# explore geographical mobility by FMA

QUERY12='''SELECT f.fma,
  round(sum(c.total*f.overlap_cbg)) AS total,
  round(sum(c.Same_house_one_yr*f.overlap_cbg)) AS same_house,
  round(sum(c.diff_house_US_one_yr*f.overlap_cbg)) AS diff_house_same_metro,
  round((sum(c.diff_house_US_one_yr*f.overlap_cbg))-(sum(c.diff_house_US_one_yr_Same_metro*f.overlap_cbg)))
       AS diff_area_total,
  (((sum(c.diff_house_US_one_yr*f.overlap_cbg))-(sum(c.diff_house_US_one_yr_Same_metro*f.overlap_cbg)))/
       (sum(c.total*f.overlap_cbg))) AS percent_diff_area,
  round(sum(c.Abroad_one_yr*f.overlap_cbg)) AS diff_country,
  ((sum(c.Abroad_one_yr*f.overlap_cbg))/(sum(c.total*f.overlap_cbg))) AS percent_diff_country,
  s.geom
FROM fmac_proportion f INNER JOIN census_geographical_mobility c
ON f.c_block = c.id2
INNER JOIN fma_shapes s
ON f.fma = s.fma
GROUP BY f.fma, s.geom
ORDER BY percent_diff_area DESC
'''


df12 = gpd_query(QUERY12)

df12[df12.columns[:-1]]

gplt.choropleth(df12,
                hue=df12['percent_diff_area'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='OrRd',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))


plt.title("Fraction of individuals that were not living in PDX metro 1 yr ago by FMA")


gplt.choropleth(df12,
                hue=df12['percent_diff_country'],  # Display data, passed as a Series
                projection=gcrs.AlbersEqualArea(),
                cmap='PuBu',
                linewidth=0.5,
                k=None,
                legend=True,
                #scheme='equal_interval',
                figsize=(10, 10))        
        
        
plt.title("Fraction of individuals that were living in another country 1 yr ago by FMA")

# try joining census_dfhousehold_income by census block group to fma to look at income distribution
# put this at the end of notebook since there are many distribution charts

QUERY13='''SELECT f.fma,
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

df13 = pquery(QUERY13)

df13.info()

df13

for i in df13['fma']:
    df13.loc[df13['fma'] == i,'less_than_10000':].plot(kind = 'bar',title = "fma#= "+str(i))
    # Shrink current axis by 20%
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
            



