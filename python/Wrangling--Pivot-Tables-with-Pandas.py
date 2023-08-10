get_ipython().magic('matplotlib inline')
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
DATA_PATH = Path('data', 'sf', 'raw', 'bulk-data.sfgov.org--20060101-20160401--cuks-n6tp.csv')

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df['year'] = df['date'].dt.year
# filter out 2016 since it's not a full year
df = df[df['year'] < 2016]

# total number of rows
len(df)

df.head()

crimeagg = df.pivot_table(index='year', aggfunc=len, values='pdid')
# the result
crimeagg

# let's chart it
fig, ax = plt.subplots()
ax.bar(crimeagg.index, crimeagg.values, align='center')
ax.set_xticks(range(2006, 2016))
ax.set_xlim(xmin=2005);

catdf = df.pivot_table(index='category', columns='year', values='pdid', aggfunc=len)

catdf.head()

catdf['delta_2015_2010'] = (catdf[2015] - catdf[2010]) / catdf[2010]
catdf.sort_values('delta_2015_2010')

# filter the list to types of crime that have had more than a 1000 reported incidents
# total for 2010 and 2015
majorcatdf = catdf[catdf[2015] + catdf[2010] > 1000]
majorcatdf.sort_values('delta_2015_2010')

thefts_df = df[df['category'] == 'LARCENY/THEFT']
thefts_pivot = thefts_df.pivot_table(index='descript', columns='year', values='pdid', aggfunc=len)

thefts_pivot['delta_2015_2010'] = (thefts_pivot[2015] - thefts_pivot[2010]) / thefts_pivot[2010] 

thefts_pivot.sort_values('delta_2015_2010').head()

# remove noise
majorthefts_pivot = thefts_pivot[thefts_pivot[2015] + thefts_pivot[2010] > 100]
majorthefts_pivot.sort_values('delta_2015_2010')

autothefts_df = df[df['category'] == 'VEHICLE THEFT']
autothefts_piv = autothefts_df.pivot_table(index='descript', columns='year', values='pdid', aggfunc=len)
autothefts_piv['delta_2015_2010'] = (autothefts_piv[2015] - autothefts_piv[2010]) / autothefts_piv[2010]
major_autothefts_piv = autothefts_piv[autothefts_piv[2015] + autothefts_piv[2010] > 100]

major_autothefts_piv.sort_values('delta_2015_2010')

v_df = df[df['category'] == 'VEHICLE THEFT']
v_piv = v_df.pivot_table(index='pddistrict', columns='year', values='pdid', aggfunc=len)
v_piv['delta_2014_2010'] = (v_piv[2014] - v_piv[2010]) / v_piv[2010]

v_piv.sort_values('delta_2014_2010', ascending=False)



