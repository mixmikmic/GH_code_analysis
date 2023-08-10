import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14.0, 5.0)
matplotlib.rcParams['axes.titlesize'] = 18

change1617 = pd.read_csv('output/changes-1617.csv')
change1617.head()

change1617.hist(column='percentChange', by='MetalLevel', figsize=(15, 15))

sortedAges = np.sort(change1617.Age.unique())
g = sns.FacetGrid(data=change1617, row='MetalLevel',  size=5, aspect=4, sharex=False)
g.map(sns.barplot, 'Age', 'percentChange', order=list(sortedAges))

states = np.sort(change1617.StateCode.unique())
g = sns.FacetGrid(data=change1617, row='MetalLevel',  size=5, aspect=4, sharex=False)
g.map(sns.barplot, 'StateCode', 'percentChange', order=list(states))

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.stripplot(data=change1617, x='StateCode', y='percentChange', hue='MetalLevel', size=8)

import json
import folium
from folium.colormap import linear

us_states =  r'us-states.json'

geo_json_data = json.load(open(us_states))

fips = pd.read_csv('fips-codes.csv', header=None)
states = pd.DataFrame({'StateCode':pd.Series(fips[0].rename(columns={0:'StateCode'}).unique())})

def makeStateMap(df, dataCol):
    m = folium.Map(location=[43, -100], zoom_start=4)
    
    dataDict = df[df[dataCol].notnull()].set_index('StateCode')[dataCol]
    
    colormap = linear.OrRd.scale(
        df.percentChange.min(),
        df.percentChange.max())
    
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

df = change1617[change1617.MetalLevel == "Gold"].groupby('StateCode').percentChange.mean().reset_index()
m = makeStateMap(df, 'percentChange')
m.save('change_1617_gold_plans.html')
m

df = change1617[change1617.MetalLevel == "Silver"].groupby('StateCode').percentChange.mean().reset_index()
m = makeStateMap(df, 'percentChange')
m.save('change_1617_silver_plans.html')
m

df = change1617[change1617.MetalLevel == "Bronze"].groupby('StateCode').percentChange.mean().reset_index()
m = makeStateMap(df, 'percentChange')
m.save('change_1617_bronze_plans.html')
m

df = change1617[change1617.MetalLevel == "Catastrophic"].groupby('StateCode').percentChange.mean().reset_index()
m = makeStateMap(df, 'percentChange')
m.save('change_1617_catastrophic_plans.html')
m

