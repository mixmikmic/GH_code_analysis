import urllib2
import re

urls = ['https://www.census.gov/did/www/saipe/downloads/estmod93/est93ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod95/est95ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod97/est97ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod98/est98ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod99/est99ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod00/est00ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod01/est01ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod02/est02ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod03/est03ALL.dat',
        'https://www.census.gov/did/www/saipe/downloads/estmod04/est04ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod05/est05ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod06/est06ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod07/est07ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod08/est08ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod09/est09ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod10/est10ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod11/est11all.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod12/est12ALL.txt',
        'https://www.census.gov/did/www/saipe/downloads/estmod13/est13ALL.txt']

State_FIPS = []
County_FIPS = []
Poverty_Estimate_All_Ages = []
Poverty_Percent_All_Ages = []
Poverty_Estimate_Under_Age_18 = []
Poverty_Percent_Under_Age_18 = []
Poverty_Estimate_Ages_5_17 = []
Poverty_Percent_Ages_5_17 = []
Median_Household_Income = []
Name = []
Postal = []
URL = []

def getUrl(urls):
    for url in urls:
        response = urllib2.urlopen(url)
        lines = response.read().split('\n')
        del lines[-1]
        for i in lines:
            State_FIPS.append(i[0:2])
            County_FIPS.append(i[3:6])
            Poverty_Estimate_All_Ages.append(i[7:15])
            Poverty_Percent_All_Ages.append(i[34:38])
            Poverty_Estimate_Under_Age_18.append(i[49:57])
            Poverty_Percent_Under_Age_18.append(i[76:80])
            Poverty_Estimate_Ages_5_17.append(i[91:99])
            Poverty_Percent_Ages_5_17.append(i[118:122])
            Median_Household_Income.append(i[133:139])
            Name.append(i[193:238])
            Postal.append(i[239:241])
            URL.append(url)
                 
getUrl(urls)

import datetime as dt
import numpy as np
import pandas as pd

data = pd.DataFrame()
data['State FIPS'] = State_FIPS
data['County FIPS'] = County_FIPS
data['County FIPS'] = data['County FIPS'].str.lstrip(' ')
data['County FIPS'] = data['County FIPS'].str.zfill(3)
data['FIPS'] = data['State FIPS']+data['County FIPS']

data['Poverty Estimate All Ages'] = Poverty_Estimate_All_Ages
data['Poverty Estimate All Ages'] = data['Poverty Estimate All Ages'].apply(lambda x: re.sub("[^0-9]","", x))
#data['Poverty Estimate All Ages'] = data['Poverty Estimate All Ages'].apply(lambda x: int(x))
data['Poverty Estimate All Ages'] = pd.to_numeric(data['Poverty Estimate All Ages'], errors='coerce')
#Check with others if have empty row ie. ''

data['Poverty Percent All Ages'] = Poverty_Percent_All_Ages
data['Poverty Percent All Ages'] = pd.to_numeric(data['Poverty Percent All Ages'], errors='coerce')

data['Poverty Estimate Under Age 18'] = Poverty_Estimate_Under_Age_18
data['Poverty Estimate Under Age 18'] = pd.to_numeric(data['Poverty Estimate Under Age 18'], errors='coerce')

data['Poverty Percent Under Age 18'] = Poverty_Percent_Under_Age_18
data['Poverty Percent Under Age 18'] = pd.to_numeric(data['Poverty Percent Under Age 18'], errors='coerce')

data['Poverty Estimate Ages 5-17'] = Poverty_Estimate_Ages_5_17
data['Poverty Estimate Ages 5-17'] = pd.to_numeric(data['Poverty Estimate Ages 5-17'], errors='coerce')

data['Poverty Percent Under Age 5-17'] = Poverty_Percent_Ages_5_17
data['Poverty Percent Under Age 5-17'] = pd.to_numeric(data['Poverty Percent Under Age 5-17'], errors='coerce')

data['Median Household Income'] = Median_Household_Income
data['Median Household Income'] = pd.to_numeric(data['Median Household Income'], errors='coerce')

data['Name'] = Name
data['Postal'] = Postal
data['url'] = URL
data['year'] = data['url'].str[-9:-7]
data['year'] = data['year'].apply(lambda x: '19'+str(x) if int(x)>50 else '20'+str(x))
data['year'] = pd.to_datetime(data['year'])
data['year'] = data['year'].dt.year

print 'Complete dataset for data collected from all of the URLs"'
print data.shape
#Check datatypes of each column
cols = data.columns
print 'Check the datatypes of each column:'
for i in cols:
    print 'The datatype for %s is' %(i), type(data[i][0])

data.head()

US_stat = data[data['FIPS'] == '00000']
US_stat = US_stat.set_index('year')
del US_stat['url']
US_stat.head()

states_stat = data[(data['County FIPS'] == '000') & (data['State FIPS'] != '00')]
states_stat = states_stat.sort_values(by=['Postal', 'year'])
states_stat = states_stat.set_index(['Postal', 'year'])
del states_stat['url']
states_stat.head()

county_stat = data[(data['State FIPS'] != '00') & (data['County FIPS'] != '000')]
county_stat = county_stat.sort_values(by=['Postal', 'FIPS', 'year'])
county_stat = county_stat.set_index(['Postal', 'FIPS', 'year'])
del county_stat['url']
county_stat.head()

import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')

US_poverty_change = float(US_stat['Poverty Percent All Ages'][US_stat.index == 2013])/float(US_stat['Poverty Percent All Ages'][US_stat.index == 2000])

print 'poverty percent change ratio from 2000 to 2013 =', US_poverty_change

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(8,3))

ax1.plot(US_stat.index, US_stat['Median Household Income'])
ax1.set_title('Median Household Income')
ax1.grid(True)
ax2.plot(US_stat.index, US_stat['Poverty Percent All Ages'])
ax2.set_title('Poverty Percent All Ages')
ax2.grid(True)

fig, axs = plt.subplots(nrows = 11, ncols=5, sharex=True, sharey=True, figsize=(20,40))
axs = axs.ravel()
i = 0
for state in (states_stat.index.get_level_values('Postal')).unique():
    A = states_stat.iloc[states_stat.index.get_level_values('Postal') == state]
    A = A.reset_index(level=0, drop=True)
    axs[i].plot(A.index, A['Poverty Percent All Ages'], label='State')
    axs[i].plot(US_stat.index, US_stat['Poverty Percent All Ages'], label='US')
    axs[i].set_title(state)
    axs[i].legend(loc='upper center', ncol=2)
    axs[i].grid(True)
    i += 1    
plt.gcf().autofmt_xdate()

#!python -m pip install --upgrade pip
#!pip install vincent

A = county_stat.iloc[(county_stat.index.get_level_values('year') == 2000) | (county_stat.index.get_level_values('year') == 2013)]
FIPS = pd.DataFrame()
FIPS['Poverty Percent All Ages'] = A['Poverty Percent All Ages']
FIPS['Name'] = A['Name']
FIPS['State'] = A.index.get_level_values('Postal')
FIPS = FIPS.unstack('year')
FIPS.columns = FIPS.columns.droplevel()
A = FIPS.iloc[:, 0:2]
A = A.rename(columns={2000: '2000', 2013:'2013'})
B = pd.DataFrame(FIPS.iloc[:, 3])
B = B.rename(columns={2013:'County'})
C = pd.DataFrame(FIPS.iloc[:, 5])
C = C.rename(columns={2013: 'State'})
FIPS = pd.concat([A, B, C], axis=1)
FIPS['change ratio'] = FIPS['2013']/FIPS['2000']
FIPS['normalized change ratio'] = FIPS['change ratio']/US_poverty_change
FIPS = FIPS.reset_index(level=0, drop=True)
FIPS['FIPS'] = FIPS.index
FIPS = FIPS.reset_index(level=0, drop=True)
FIPS['FIPS'] = FIPS['FIPS'].astype(int)
print FIPS.shape
FIPS.head()

#To initialize the map; create a sample
import vincent
vincent.core.initialize_notebook()
import json

county_topo = r'us_counties.topo.json'
geo_data = [{'name': 'counties',
         'url': county_topo,
         'feature': 'us_counties.geo'}]
vis = vincent.Map(geo_data=geo_data, scale=1000, projection='albersUsa')
vis

#Referenced the following link: http://wrobstory.github.io/2013/10/mapping-data-python.html
import json

#Map the county code we have in our geometry to those in the FIPS file
with open('us_counties.topo.json', 'r') as f:
    get_id = json.load(f)

#A little FIPS code type casting to ensure key match
new_geoms = []
for geom in get_id['objects']['us_counties.geo']['geometries']:
    geom['properties']['FIPS'] = int(geom['properties']['FIPS'])
    new_geoms.append(geom)
    
get_id['objects']['us_counties.geo']['geometries'] = new_geoms

with open('us_counties.topo.json', 'w') as f:
    json.dump(get_id, f)

geometries = get_id['objects']['us_counties.geo']['geometries']
county_codes = [x['properties']['FIPS'] for x in geometries]
county_df = pd.DataFrame({'FIPS': county_codes}, dtype=str)
county_df = county_df.astype(int)
print county_df.shape

#Perform an inner join, pad NA's with data from nearest county
merged = pd.merge(FIPS, county_df, on='FIPS', how="inner")
merged = merged.fillna(method='pad')
print merged.shape
merged.head()

import vincent
vincent.core.initialize_notebook()

county_topo = r'us_counties.topo.json'
geo_data = [{'name': 'counties',
            'url': county_topo,
            'feature': 'us_counties.geo'}]

vis = vincent.Map(data=merged, geo_data=geo_data, scale=1100,
                  projection='albersUsa', data_bind='2000',
                  data_key='FIPS', map_key={'counties': 'properties.FIPS'})
vis.scales['color'].type = 'threshold'
vis.scales['color'].domain = [0, 4, 6, 8, 10, 12, 20, 30]
vis.legend(title='Poverty 2000 (%)')
# vis.to_json('graph1.json')
vis

county_topo = r'us_counties.topo.json'
geo_data = [{'name': 'counties',
            'url': county_topo,
            'feature': 'us_counties.geo'}]

vis1 = vincent.Map(data=merged, geo_data=geo_data, scale=1100,
                  projection='albersUsa', data_bind='2013',
                  data_key='FIPS', map_key={'counties': 'properties.FIPS'})
vis1.scales['color'].type = 'threshold'
vis1.scales['color'].domain = [0, 4, 6, 8, 10, 12, 20, 30]
vis1.legend(title='Poverty 2013 (%)')
# vis.to_json('graph1.json')
vis1

county_topo = r'us_counties.topo.json'
geo_data = [{'name': 'counties',
            'url': county_topo,
            'feature': 'us_counties.geo'}]

vis2 = vincent.Map(data=merged, geo_data=geo_data, scale=1100,
                  projection='albersUsa', data_bind='normalized change ratio',
                  data_key='FIPS', map_key={'counties': 'properties.FIPS'})
vis2.scales['color'].type = 'threshold'
vis2.scales['color'].domain = [0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
vis2.legend(title='Change in Poverty (%)')
# vis.to_json('graph1.json')
vis2

top = merged.sort_values(by = ['normalized change ratio'], ascending=False).head(1)
low = merged.sort_values(by = ['normalized change ratio'], ascending=True).head(1)

print 'County with max increase in poverty --', top.iloc[0]['County'], 'in state', top.iloc[0]['State'], 'change=', top.iloc[0]['normalized change ratio']
print 'County with max decrease in poverty --', low.iloc[0]['County'], 'in state', low.iloc[0]['State'], 'change=', low.iloc[0]['normalized change ratio']



