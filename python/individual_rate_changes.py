import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14.0, 5.0)
matplotlib.rcParams['axes.titlesize'] = 18

change1415 = pd.read_csv('change1415.csv')
change1516 = pd.read_csv('change1516.csv')
change1617 = pd.read_csv('change1617.csv')
change1415.head()

change1617[(change1617.Age == '25') & (change1617.StateCode == "SC")]['avg(IndividualRate)_2017']

import re
toMatch = re.compile('\(\w+\)')
renameCols = {col:  col.replace('(', "").replace(')', "").replace('avg', "") for col in change1415.columns  if 'change' in col}
change1415 = change1415.rename(columns=renameCols)
change1516 = change1516.rename(columns=renameCols)
change1617 = change1617.rename(columns=renameCols)

change1415 = change1415[change1415.Age != 'Family Option']
change1516 = change1516[change1516.Age != 'Family Option']
change1617 = change1617[change1617.Age != 'Family Option']

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(16,18))

change1415.groupby(['Age'])['change_IndividualRate'].mean().plot.bar(ax=ax1)
ax1.set_title('% Change from 2014 to 2015 in Individual Rates, by Age')
ax1.set_ylim((-25,25))

change1516.groupby(['Age'])['change_IndividualRate'].mean().plot.bar(ax=ax2)
ax2.set_title('% Change from 2015 to 2016 in Individual Rates, by Age')
ax2.set_ylim((-25,25))

change1617.groupby(['Age'])['change_IndividualRate'].mean().plot.bar(ax=ax3)
ax3.set_title('% Change from 2016 to 2017 in Individual Rates, by Age')
ax3.set_ylim((-25,25))

fig.tight_layout()

change1415['years'] = '2014-2015'
change1516['years'] = '2015-2016'
change1617['years'] = '2016-2017'

allYears = pd.concat([change1415, change1516, change1617])

g = sns.FacetGrid(row='years', data=allYears[allYears.Age != 'Family Option'], size=8, aspect=2.5, sharex=False)
g.map(sns.barplot,'StateCode','change_IndividualRate')

def binAge(age):
    if age.find('-') > -1:
        return '0-30'
    if age.find('and') > -1:
        return '55+'
    if age == 'Family Option':
        return 'Family Option'
    
    age = int(age)
    
    if age < 31:
        return '0-30'
    
    if age > 31 and age < 55:
        return '31-54'
    
    return '55+'

change1415['ageBin'] = change1415.Age.apply(binAge)
change1516['ageBin'] = change1516.Age.apply(binAge)
change1617['ageBin'] = change1617.Age.apply(binAge)

order = list(change1415.ageBin.unique())
fig, ax =plt.subplots(1,1, figsize=(20, 25))
sns.barplot(y='StateCode', x='change_IndividualRate', hue='ageBin', ci=None, hue_order=order, data=change1415)
ax.set_title('Rate changes by state and age, 2014-2015')
ax.set_xlim([-100, 100])

fig, ax =plt.subplots(1,1, figsize=(20, 25))
sns.barplot(y='StateCode', x='change_IndividualRate', hue='ageBin',hue_order=order, ci=None, data=change1516)
ax.set_title('Rate changes by state and age, 2015-2016')
ax.set_xlim([-100, 100])

fig, ax =plt.subplots(1,1, figsize=(20, 25))
sns.barplot(y='StateCode', x='change_IndividualRate', hue='ageBin',hue_order=order, ci=None, data=change1617)
ax.set_title('Rate changes by state and age, 2016-2017')
ax.set_xlim([-100, 100])

import json
from folium.colormap import linear

us_states =  r'us-states.json'

geo_json_data = json.load(open(us_states))

fips = pd.read_csv('fips-codes.csv', header=None)
states = pd.DataFrame({'StateCode':pd.Series(fips[0].rename(columns={0:'StateCode'}).unique())})

def makeStateMap(df, dataCol):
    m = folium.Map(location=[43, -100], zoom_start=4)
    
    dataDict = df[df[dataCol].notnull()].set_index('StateCode')[dataCol]
    
    colormap = linear.RdBu.scale(
        -25,
        25)
    
    #we need to all the states in our dataset, or folium will throw an error. So we add the states that are missing
    statesToAppend =  pd.Series(index=[state for state in states.StateCode if state not in dataDict.index.values])
    
    #determine the colors for each state -- note we use black for 
    #states that were originally missing from our dataset 
    dataDict = dataDict.append(statesToAppend).fillna(-9999)
    color_dict = {key: colormap(dataDict[key]) if dataDict[key] != -9999 else '#000' for key in dataDict.keys()}
    
    folium.GeoJson(
    geo_json_data,
    style_function=lambda feature: {
        'fillColor': color_dict[feature['id']],
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.9,
    }
).add_to(m)
    
    colormap.caption = 'Percentage Change'
    colormap.add_to(m)
    return m

df = change1415.groupby("StateCode").change_IndividualRate.mean().reset_index()
m = makeStateMap(df, 'change_IndividualRate')
m

df = change1516.groupby("StateCode").change_IndividualRate.mean().reset_index()
m = makeStateMap(df, 'change_IndividualRate')
m

df = change1617.groupby("StateCode").change_IndividualRate.mean().reset_index()
m = makeStateMap(df, 'change_IndividualRate')
m



