import pandas as pd

df = pd.read_csv('http://www.bls.gov/lau/laucntycur14.txt', header=None, skiprows=6, sep='|',
                names=['LAUS','FIPS_STATE','FIPS_COUNTY','AREA_TITLE','PERIOD','CIV_LABOR_FORCE','EMPLOYED',
                       'UNEMPL_LEVEL','UNEMPL_RATE'],
                dtype={'FIPS_STATE':str, 'FIPS_COUNTY':str})

df.head()

df['FIPS_STATE'] = df['FIPS_STATE'].str.strip()         # remove/strip leading or trailing whitespaces
df['FIPS_STATE'] = df['FIPS_STATE'].str.lstrip('0')     # left strip / remove leading zeros
df['FIPS_COUNTY'] = df['FIPS_COUNTY'].str.strip()

df['region'] = df.FIPS_STATE + df.FIPS_COUNTY           # concatenate or mesh together the state and county FIPS code
df['value'] = df['UNEMPL_RATE']                         # create a column called "value" containing unemployment rates

df.head()                                               # Let's see what the first 5 rows of our data looks like

df.PERIOD.value_counts()

criteria1 = df['PERIOD'].str.contains('Mar-15', na=False)
criteria2 = df['FIPS_STATE'] != '72'
recent = df[criteria1 & criteria2]

recent.head()

recent.PERIOD.value_counts()

choropleth_data = recent[['region','value','AREA_TITLE']]

choropleth_data.head()

# load the rpy2 extension
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

get_ipython().run_line_magic('R', '-i choropleth_data')

get_ipython().run_cell_magic('R', '-w 800 -h 500 -u px', 'library(choroplethr)\nchoropleth_data = as.data.frame(choropleth_data)\ncounty_choropleth(choropleth_data)')

get_ipython().run_cell_magic('R', '-w 800 -h 500 -u px', 'library(choroplethr)\nchoropleth_data = as.data.frame(choropleth_data)\nchoropleth_data$value = as.numeric(as.character(choropleth_data$value))\ncounty_choropleth(choropleth_data, title = "Unemployment Rate by US Counties (March 2015)",\n                 legend = "Unemployment Rate")')

