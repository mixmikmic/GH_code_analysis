import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_url = "http://www.spc.noaa.gov/wcm/data/"

years = ["2010","2011","2012","2013","2014","2015","2016"]
types = ['torn','hail','wind']

source_headings = {
    'torn':['om', 'yr', 'mo', 'dy', 'date', 'time', 'tz', 'st', 'stf', 'stn', 'mag',
           'inj', 'fat', 'loss', 'closs', 'slat', 'slon', 'elat', 'elon', 'len',
           'wid', 'ns', 'sn', 'sg', 'f1', 'f2', 'f3', 'f4', 'fc'],
    'hail':['om','yr','mo','dy','date','time','tz','st','stf','stn','mag',
            'inj','fat','loss','closs','slat','slon','elat','elon','len',
            'wid','ns','sn','sg','f1','f2','f3','f4'],
    'wind':['om','yr','mo','dy','date','time','tz','st','stf','stn','mag',
            'inj','fat','loss','closs','slat','slon','elat','elon','len',
            'wid','ns','sn','sg','f1','f2','f3','f4','mt']
}

filtered_columns = ["yr","mo","dy","date","time","st","mag","inj","fat","loss","closs","slat","slon"]

entire_df = pd.DataFrame()

for year in years:
    for weather_type in types:
        url = base_url+''+year+'_'+weather_type+'.csv'
        df = pd.read_csv(url, header=None)
        df.columns = source_headings[weather_type]
        df = df[filtered_columns]
        df = df.set_index('yr')
        df = df.drop(['yr'])
        df = df.reset_index()
        df['type'] = weather_type
        entire_df = entire_df.append(df)
        print(df.head())

entire_df.shape

entire_df = entire_df.reset_index(drop=True)

entire_df = entire_df.reset_index()

entire_df['yr'] = pd.to_numeric(entire_df['yr'],errors="coerce")
entire_df['mo'] = pd.to_numeric(entire_df['mo'],errors="coerce")
entire_df['dy'] = pd.to_numeric(entire_df['dy'],errors="coerce")
entire_df['mag'] = pd.to_numeric(entire_df['mag'],errors="coerce")
entire_df['inj'] = pd.to_numeric(entire_df['inj'],errors="coerce")
entire_df['fat'] = pd.to_numeric(entire_df['fat'],errors="coerce")
entire_df['loss'] = pd.to_numeric(entire_df['loss'],errors="coerce")
entire_df['closs'] = pd.to_numeric(entire_df['closs'],errors="coerce")
entire_df['slat'] = pd.to_numeric(entire_df['slat'],errors="coerce")
entire_df['slon'] = pd.to_numeric(entire_df['slon'],errors="coerce")

entire_df['date_time'] = entire_df['date']+' '+entire_df['time']


entire_df['date_time'] = pd.to_datetime(entire_df['date_time'],format="%Y-%m-%d %H:%M:%S")
    
entire_df.dtypes

entire_df = entire_df.drop(['date','time'], axis=1)

entire_df.head()

# none to drop

entire_df.isnull().sum()

entire_df.to_csv('AllEvents.csv')

Events = 'AllEvents.csv'
AllEvents_df = pd.read_csv(Events)

AllEvents_df.head()

Events = AllEvents_df.groupby('type')

count_Events = Events['yr'].count()

count_Events

Events = AllEvents_df.groupby('yr')

count_Events = Events['type'].count()

count_Events

# Labels for the sections of our pie chart
labels = ["hail", "torn", "wind"]

# The values of each section of the pie chart
sizes = sizes = [15, 30, 45]

# The colors of each section of the pie chart
colors = ["yellowgreen", "red", "lightskyblue"]

# Tells matplotlib to seperate the "Python" section from the others
explode = (0.1, 0, 0)

# Creates the pie chart based upon the values above
# Automatically finds the percentages of each part of the pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=140)

plt.axis("equal")

plt.show()

gyms = ["inj", "fat", "closs", "loss"]
members = [49, 92, 84, 53]
colors = ["yellowgreen", "red", "purple", "lightskyblue"]
explode = (0, 0.05, 0, 0)

plt.title("Severe Events")
plt.pie(members, explode=explode, labels=gyms, colors=colors,
        autopct="%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.show()





