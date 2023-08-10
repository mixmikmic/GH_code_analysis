from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')

commits = pd.read_csv("/Users/catherineordun/Documents/Hackathon/Data/commits_fin2.csv")

commits.head()

commits.isnull().sum()

#virginia state only 

import json
import branca

virginia = r'/Users/catherineordun/Documents/Data/virginia_county_geofile.json'

type(virginia)

import os
import folium
print(folium.__version__)
import json
import geopandas

gdf = geopandas.read_file(virginia)

m = folium.Map([37.54, -77.43], zoom_start=7, tiles='cartodbpositron')

folium.GeoJson(
    gdf,
).add_to(m)

#m.save(os.path.join('results', 'GeoJSON_and_choropleth_3.html'))

m

gdf.head()

commits.sort_values(by='FIPS Code')

gdf['geoid'][:10]

gdf['FIPS Code'] = gdf['geoid'].astype(str).str[-5:].astype(np.int64)
gdf['FIPS Code'][:10]

commits.dtypes

len(commits)

gdf_new = pd.merge(gdf, commits, on='FIPS Code', how='left')

gdf_new

len(gdf_new)

#For year 2010
df2010 = gdf_new.loc[(gdf_new['Year'] == 2010.0)]
print(len(df2010))

#For year 2011
df2011 = gdf_new.loc[(gdf_new['Year'] == 2011.0)]
print(len(df2011))

#For year 2012
df2012 = gdf_new.loc[(gdf_new['Year'] == 2012.0)]
print(len(df2012))

#For year 2013
df2013 = gdf_new.loc[(gdf_new['Year'] == 2013.0)]
print(len(df2013))

#For year 2014
df2014 = gdf_new.loc[(gdf_new['Year'] == 2014.0)]
print(len(df2014))

#For year 2015
df2015 = gdf_new.loc[(gdf_new['Year'] == 2015.0)]
print(len(df2015))

#For year 2016
#no data for 2016
df2016 = gdf_new.loc[(gdf_new['Year'] == 2016.0)]
print(len(df2016))

#For year 2017
#no data for 2017
df2017 = gdf_new.loc[(gdf_new['Year'] == 2017.0)]
print(len(df2017))

df2015.head()

# plot
sns.set(context='poster', style='darkgrid', palette='deep')
fig, ax = plt.subplots()
# set figure size
fig.set_size_inches(35,20)
ax.set(ylim=(-1, 10))
ax = sns.boxplot(x="MSO", y="count", hue="DrugUsage", data=df2015)

# plot
sns.set(context='poster', style='darkgrid', palette='deep')
fig, ax = plt.subplots()
# set figure size
fig.set_size_inches(35,15)
ax.set(ylim=(-1, 10))
ax = sns.boxplot(x="DrugUsage", y="count", hue="MSO", data=df2015)

df2015_larceny = df2015.loc[(df2015['MSO'] == 'Larceny/Fraud') & (df2015['DrugUsage'] =='Heavy Use')]
df2015_l_ = df2015_larceny.reset_index() #needed to reset index otherwise when reading in the loop the styles will not attach becuse the older df will not be ordinal indexes
df2015_l_

df2015_l_['count'].describe()

df2015_l_['count'].hist(bins=25)

styles = []

for row in df2015_l_['count']:
    if row >= 8:
        styles.append({'fillColor': '#016c59', 'weight': 1, 'color': '#000000'})
    elif row >= 6:
        styles.append({'fillColor': '#1c9099', 'weight': 1, 'color': '#000000'})
    elif row >= 4:
        styles.append({'fillColor': '#67a9cf', 'weight': 1, 'color': '#000000'})
    elif row >= 2:
        styles.append({'fillColor': '#bdc9e1', 'weight': 1, 'color': '#000000'})
    elif row < 2:
        styles.append({'fillColor': '#f6eff7', 'weight': 1, 'color': '#000000'})

df2015_l_.head()

styles = pd.DataFrame(np.array(styles))
df2015_l_['style'] = styles

df2015_l_.head()

import pysal
print(pysal.esda.mapclassify.Map_Classifier.__doc__)

f, ax = plt.subplots(1, figsize=(18, 10))
df2015_l_.plot(column='count', scheme='fisher_jenks', k=7, 
                         alpha=0.9, cmap=plt.cm.Blues, legend=True, ax=ax)
plt.axis('equal')
ax.set_title('Count of Larceny/Fraud Cases in 2015 Virginia associated with Heavy Use of Drugs');

import branca.colormap as cm
m = folium.Map([37.54, -77.43], zoom_start=8,  tiles='Mapbox Bright')
folium.GeoJson(df2015_l_).add_to(m)

colormap = cm.linear.PuBuGn.scale(0, 12)
colormap.caption = 'Counts of Larceny/Fraud Cases with Heavy Drug Use in 2015'
m.add_child(colormap)

m

df2015_larceny_mod = df2015.loc[(df2015['MSO'] == 'Larceny/Fraud') & (df2015['DrugUsage'] =='Moderate Use')]
df2015_mod_ = df2015_larceny_mod.reset_index() #needed to reset index otherwise when reading in the loop the styles will not attach becuse the older df will not be ordinal indexes
df2015_mod_

df2015_mod_['style'] = styles

len(df2015_mod_)

len(df2015_l_)

f, ax = plt.subplots(1, figsize=(18, 10))
df2015_mod_.plot(column='count', scheme='fisher_jenks', k=4, 
                         alpha=0.9, cmap=plt.cm.Blues, legend=True, ax=ax)
plt.axis('equal')
ax.set_title('Count of Larceny/Fraud Cases in 2015 Virginia associated with MODERATE Use of Drugs');

m = folium.Map([37.54, -77.43], zoom_start=8,  tiles='Mapbox Bright')
folium.GeoJson(df2015_mod_).add_to(m)

colormap = cm.linear.PuBuGn.scale(0, 7)
colormap.caption = 'Counts of Larceny/Fraud Cases with Moderate Drug Use in 2015'
m.add_child(colormap)

m


#Read in the mortality data
mortality = pd.read_csv("/Users/catherineordun/Documents/Hackathon/Data/Elastic/op_dash_age.csv")

mortality.dtypes

#merge with gdf frame
gdf_mortality = pd.merge(gdf, mortality, on='FIPS Code', how='left')

#For year 2015
mort2015 = gdf_mortality.loc[(gdf_mortality['Year'] == 2015.0)]
print(len(mort2015))

mort2015.head()

# plot
sns.set(context='poster', style='darkgrid', palette='deep')
fig, ax = plt.subplots()
# set figure size
fig.set_size_inches(35,15)
ax.set(ylim=(-1, 500))
ax = sns.boxplot(x="Type", y="Rate", hue="Age Group", data=mort2015)

#Age Group: 15-24; Type: ED Opioid Overdose
mort2015_a = mort2015.loc[(mort2015['Age Group'] == '15-24') & (mort2015['Type'] =='ED Opioid Overdose')]
mort2015_a_ = mort2015_a.reset_index() #needed to reset index otherwise when reading in the loop the styles will not attach becuse the older df will not be ordinal indexes
mort2015_a_

f, ax = plt.subplots(1, figsize=(18, 10))
mort2015_a_.plot(column='Rate', scheme='fisher_jenks', k=5, 
                         alpha=0.9, cmap=plt.cm.Blues, legend=True, ax=ax)
plt.axis('equal')
ax.set_title('Year 2015: Age Group: 15-24; Type: ED Opioid Overdose');

#Age Group: 25-34; Type: Fatal Prescription Opioid Overdose
mort2015_b = mort2015.loc[(mort2015['Age Group'] == '25-34') & (mort2015['Type'] =='Fatal Prescription Opioid Overdose')]
mort2015_b_ = mort2015_b.reset_index() #needed to reset index otherwise when reading in the loop the styles will not attach becuse the older df will not be ordinal indexes
mort2015_b_.head()

f, ax = plt.subplots(1, figsize=(18, 10))
mort2015_b_.plot(column='Rate', scheme='fisher_jenks', k=5, 
                         alpha=0.9, cmap=plt.cm.Blues, legend=True, ax=ax)
plt.axis('equal')
ax.set_title('Year 2015: Age Group: 25-34; Type: Fatal Prescription Opioid Overdose Opioid Overdose');



#IGNORE

import json
import numpy as np
import vincent

# Let's create the vincent chart.
scatter_chart = vincent.Bar(opioids_51195['Rate'].values.tolist(), width=200,
                                height=100)

# Let's convert it to JSON.
opioids_51195_json = scatter_chart.to_json()

# Let's convert it to dict.
scatter_dict = json.loads(opioids_51195_json)

#do it for the second plot
scatter_chart2 = vincent.Bar(opioids_51087['Rate'].values.tolist(), width=200,
                                height=100)
opioids_51087_json = scatter_chart2.to_json()
scatter_dict2 = json.loads(opioids_51087_json)

import branca.colormap as cm
m = folium.Map([37.54, -77.43], zoom_start=7,  tiles='Mapbox Bright')
folium.GeoJson(op2016).add_to(m)

colormap = cm.linear.PuBuGn.scale(0, 506)
colormap.caption = 'Rate of ED Opioid Overdoses in 2016'
m.add_child(colormap)

# Let's create a Vega popup based on scatter_json - Wise County, VA FIPS 51195
popup = folium.Popup(max_width=250)
folium.Vega(opioids_51195_json, height=150, width=250).add_to(popup)
folium.Marker([37.0241, -82.6051], popup=popup).add_to(m)

# Let's create a Vega popup based on scatter_json - Henrico County VA fips 51087
popup = folium.Popup(max_width=250)
folium.Vega(opioids_51087_json, height=150, width=250).add_to(popup)
folium.Marker([37.5059, -77.3324], popup=popup).add_to(m)


m



