## Import necessary packages
import pandas as pd
import numpy as np
import folium
from folium import plugins
from folium.plugins import MarkerCluster
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Set pandas's max row display
pd.set_option('display.max_row', 1000)

# Set pandas's max column width to 50
pd.set_option('display.max_columns', 50)

# Load in the Chicago crime dataset
df = pd.read_csv('Crimes_-_2001_to_present.csv')

## Print first 5 lines of dataset
df.head()

## Print last 5 lines of dataset
df.tail()

df.columns

## Check if any rows are missing data and are null
df['Primary Type'].isnull().values.any()

## Count number of observations for each crime
df['Primary Type'].value_counts()

## Plot these for better visualization
crime_type_df = df['Primary Type'].value_counts(ascending=True)

## Some formatting to make it look nicer
fig=plt.figure(figsize=(18, 16))
plt.title("Frequency of Crimes Per Crime Type")
plt.xlabel("Frequency of Crimes")
plt.ylabel("Type of Crime")
ax = crime_type_df.plot(kind='barh')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

## Check if any rows are missing data and are null
df['Year'].isnull().values.any()

## Count number of reported crimes for each year
df['Year'].value_counts()

## Plot these for better visualization
crime_year_df = df['Year'].value_counts(ascending=True)

## Some formatting to make it look nicer
fig=plt.figure(figsize=(10, 8))
plt.title("Frequency of Crimes Per Year in Chicago")
plt.xlabel("Frequency of Crimes")
plt.ylabel("Year")
ax = crime_year_df.plot(kind='barh')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

## Check if any rows are missing data and are null
df['Arrest'].isnull().values.any()

## Count number of successful arrests for each year
df['Arrest'].value_counts()

## Convert values into percentages
arrest_df = df['Arrest'].value_counts()
arrest_percent = (arrest_df / df['Arrest'].sum()) * 100 

## Rename Series.name
arrest_percent.rename("% of Arrests",inplace=True)

## Rename True and False to % Arrested and % Not Arrested
arrest_percent.rename({True: '% Arrested', False: '% Not Arrested'},inplace=True)

## Format pie chart to nicely show percentage and count
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

## Plot results in a pie chart
arrest_percent.plot.pie(fontsize=11,
                       autopct=make_autopct(df['Arrest'].value_counts()),
                       figsize=(8, 8))

## Group dataset by year and arrests
arrest_per_year = df.groupby('Year')['Arrest'].value_counts().rename('Counts').to_frame()
arrest_per_year['Percentage'] = (100 * arrest_per_year / arrest_per_year.groupby(level=0).sum())
arrest_per_year.reset_index(level=[1],inplace=True)
arrest_per_year

## Create a line plot for percentages of successful arrests over time (2001 to present)
line_plot = arrest_per_year[arrest_per_year['Arrest'] == True]['Percentage']

## Configure line plot to make visualizing data cleaner
labels = line_plot.index.values
fig=plt.figure(figsize=(12, 10))
plt.title('Percentages of successful arrests from 2001 to 2018')
plt.xlabel("Year")
plt.ylabel("Successful Arrest Percentage")
plt.xticks(line_plot.index, line_plot.index.values)

line_plot.plot(grid=True, marker='o', color='mediumvioletred')

import datetime

## Clean data, create copy, and filter based on this month (March 2018)
cleaned_df = df[df['Latitude'].notnull() & df['Longitude'].notnull()].copy()
cleaned_df['Date Time'] = pd.to_datetime(cleaned_df['Date'], format='%m/%d/%Y %I:%M:%S %p')
cleaned_df = cleaned_df[cleaned_df['Date Time']  > datetime.datetime(2018, 2, 28)]
print("Number of crimes in Chicago since start of March 2018: %d" % len(cleaned_df))

## Create map and markers for each crime in Chicago March 2018 using folium 
crimes_map = folium.Map(location=[cleaned_df['Latitude'].mean(), cleaned_df['Longitude'].mean()], zoom_start=10)
marker_cluster = MarkerCluster().add_to(crimes_map)

#%%timeit
#for i in range(0,len(cleaned_df)):
#    popup = popup = "<p> Crime ID: " + str(cleaned_df["ID"].iloc[i]) +  "<br> Date and Time: " + cleaned_df["Date"].iloc[i] + "<br> Crime Type: " + cleaned_df["Primary Type"].iloc[i] + "<br> Crime Description: " + cleaned_df["Description"].iloc[i] + "<br> Address: " + cleaned_df["Block"].iloc[i] + "</p>"
#    folium.Marker([cleaned_df['Latitude'].iloc[i],cleaned_df['Longitude'].iloc[i]], popup=popup).add_to(marker_cluster)

#%%timeit
#for row in cleaned_df.itertuples():
#    popup = "<p> Crime ID: " + str(row[1]) +  "<br> Date and Time: " + row[3] + "<br> Crime Type: " + row[6] + "<br> Crime Description: " + row[7] + "<br> Address: " + row[4] + "</p>"
#    folium.Marker([row[20], row[21]], popup=popup).add_to(marker_cluster)

#%%timeit
#cleaned_df.apply(lambda row:folium.Marker(location=[row["Latitude"], row["Longitude"]], popup="<p> Crime ID: " + str(row["ID"]) +  "<br> Date and Time: " + row["Date"] + "<br> Crime Type: " + row["Primary Type"] + "<br> Crime Description: " + row["Description"] + "<br> Address: " + row["Block"] + "</p>").add_to(marker_cluster), axis=1)

## Winner of the 4 chosen loop techniques (very close though)
## Show first 750 values so folium map can properly render inline in Google Chrome
for row in cleaned_df.values[:750]:
    popup = "<p> Crime ID: " + str(row[0]) +  "<br> Date and Time: " + row[2] + "<br> Crime Type: " + row[5] + "<br> Crime Description: " + row[6] + "<br> Address: " + row[3] + "</p>"
    folium.Marker([row[19], row[20]], popup=popup).add_to(marker_cluster)
#crimes_map.save('March-2018-chicago-crimes.html')
crimes_map

## Create heat map of crimes in Chicago March 2018
crimes_heatmap = folium.Map(location=[cleaned_df['Latitude'].mean(), cleaned_df['Longitude'].mean()], zoom_start=10)
crimes_heatmap.add_child(plugins.HeatMap([[row[20], row[21]] for row in cleaned_df.itertuples()]))
crimes_heatmap.save("March-2018-chicago-crime-heatmap.html")
crimes_heatmap

# calculating total number of incidents per district
district_crime = df['District'].value_counts(ascending=True)

## Data cleaning
district_crime.index = district_crime.index.astype(int)
district_crime.index = district_crime.index.astype(str)

## Plot bar graph for initial visualization
fig=plt.figure(figsize=(14, 12))
plt.title("Frequency of Crimes Per Chicago Police District")
plt.xlabel("Frequency of Crimes")
plt.ylabel("Chicago Police District No.")
ax = district_crime.plot(kind='barh')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

## Reset index and name the district and crime count columns
district_crime = district_crime.reset_index()
district_crime.columns = ['District', 'Count']

# creation of the choropleth
geo_path = 'data/Boundaries - Police Districts (current).geojson'
district_map = folium.Map(location=[cleaned_df['Latitude'].mean(), cleaned_df['Longitude'].mean()], zoom_start=10)
threshold_scale = list(np.linspace(0,450000,6))
district_map.choropleth(geo_data=geo_path,
              name='choropleth',
              data = district_crime,
              threshold_scale=threshold_scale,
              columns = ['District', 'Count'],
              key_on = 'feature.properties.dist_num',
              fill_color = 'YlOrRd',
              fill_opacity = 0.7,
              line_opacity = 0.2,
              legend_name = 'Frequency of crimes per district',
              highlight=True)


district_map.save("Crime-per-district-choropleth.html")
district_map



