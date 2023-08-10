import pandas as pd
get_ipython().magic('matplotlib inline')

df = pd.read_csv('wind.csv')

df.head()

df.info()

df.tail()

df_2015 = df[df['Year'] == 2015]
df_2015.head()

df_2015_grouped = df_2015.groupby('Month')
df_2015_grouped.head(2) # Top 2 rows in each DataFrame in the groupby object

df_2015_grouped_temp = df_2015_grouped['Temperature']
df_2015_grouped_temp.head(2) # Top 2 rows of the Temperature column in the groupby object

df_2015_grouped_temp_mean = df_2015_grouped_temp.mean()
df_2015_grouped_temp_mean

df_2015_grouped_temp_mean.plot(kind='bar',title='Average Monthy Temperature (Celsius) in 2015')

df[df['Year'] == 2015].groupby('Month')['Temperature'].mean().plot(kind='bar',title='Average Monthy Temperature (Celsius) in 2015')

df[df['Month'] == 1].groupby('Wind Direction').size().plot(kind='bar',title='Hourly Wind Direction Measurements in January (2010-2016)')

df[df['Month'] == 8].groupby('Wind Direction').size().plot(kind='bar',title='Hourly Wind Direction Measurements in August (2010-2016)')

df.groupby('Wind Direction').size().plot(kind='bar',title='Hourly Wind Direction Measurements (2010-2016)')

df_March_2016 = df[(df['Month'] == 3) & (df['Year'] == 2016)]
df_March_2016.groupby('Hour')['Wind Speed'].mean().plot(kind='bar',title='Average Wind Speed (km/h) by Hour in March 2016')

df[(df['Year'] == 2016) & (df['Month'] == 3)].groupby('Hour')['Temperature'].mean().plot(kind='bar')

df[(df['Year'] == 2015) & (df['Month'] == 12)]['Wind Speed'].hist(bins=15)

