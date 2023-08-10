import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

# Read the file into data frame 
get_ipython().magic("time df_all= pd.read_csv('dublin_2012_11_distance.csv',dtype={ 'Journey_Pattern_ID': object})")

df_all.head()

df=df_all

# Only keep those At_Stop == 1
df=df.drop(df.index[df['At_Stop']==0])

# Get the mean distance of every Stop for every Journey_Pattern_ID
zscore=lambda x: x.mean()
df['Distance']=df.groupby(['Journey_Pattern_ID','Stop_ID'])['Distance'].transform(zscore)

df[['Journey_Pattern_ID','Stop_ID']]

df.head()

# Get those columns for 
df_stops=df[['Journey_Pattern_ID','Stop_ID','Distance']]

# Drop dubplicates 
df_stops=df_stops.drop_duplicates()

# Check Journey_Pattern_ID  unique value 
df['Journey_Pattern_ID'].unique()

# Drop strange Journey_Pattern_ID 
df_stops.drop(df_stops.index[df_stops['Journey_Pattern_ID']=='OL77X101'],inplace=True)
df_stops.drop(df_stops.index[df_stops['Journey_Pattern_ID']=='PP071001'],inplace=True)
# Drop by the value of one feature 
#df=df.drop(df.index[df['Journey_Pattern_ID'].isnull()])

# Change Journey_Pattern_ID type to string 
df_stops['Journey_Pattern_ID']=df_stops['Journey_Pattern_ID'].astype(str)

# Get the Line_ID and Direction from Journey_Pattern_ID 
df_stops['Line_ID']= df_stops['Journey_Pattern_ID'].apply(lambda x:x[1:4:])
df_stops['Direction']= df_stops['Journey_Pattern_ID'].apply(lambda x:x[5:6])

# Check size of the data frame 
df_stops.shape

df_stops.dtypes

# sort the data frame by Journey_Pattern_ID  and Distance 
df_stops.sort(['Journey_Pattern_ID','Distance'])

# Save the data frame to table 
df_stops.to_csv('journey_stops.csv',index=False)

# Groupby the data frame by 'Journey_Pattern_ID','Vehicle_Journey_ID','Date' and get the first row 
df_groupby=df.groupby(['Journey_Pattern_ID','Vehicle_Journey_ID','Date'])
df_first_stop=df_groupby.first()

#Reset the index for the data frame 
df_first_stop.reset_index(inplace=True)

df_first_stop.head()

# Get all the columns for the timetable 
df_timetable=df_first_stop[['Journey_Pattern_ID','Timestamp','day_of_week','datetime','First_Stop','Last_Stop']]

# Add midweek feature 
df_timetable['week_cate'] = df_timetable['day_of_week'].map({'Monday': 'Workday', 'Tuesday': 'Workday','Wednesday':'Workday','Thursday':'Workday','Friday': 'Workday','Saturday':'Saturday','Sunday':'Sunday'})

# Round the timestamp by 5 mins and add it to the new column Time 
ns_five=5*60
df_timetable['Time']=pd.to_datetime(((df_timetable['Timestamp'] // ns_five + 1 ) * ns_five),unit='s')

# Get the time from column Time we got before 
df_timetable['Time']=df_timetable['Time'].dt.time

# Sort data frame by Journey_Pattern_ID
df_timetable.sort('Journey_Pattern_ID')

import numba 

@numba.jit
def delete_error(df):
# Write the function to check if the bus start time appears more than twice, if it is, keep the time 
# Those bus start times only appear once should be error, so we will not keep them  

    df_grouped = df.groupby(['Journey_Pattern_ID','Time'])
    vj_groups = []
    for jer_name, jer_group in df_grouped:
        number=jer_group.shape[0]
        #print(number)
        if number>1:       
            vj_groups.append(jer_group )
    return pd.concat(vj_groups)
df_timetable=delete_error(df_timetable)

# Check data frame size 
df_timetable.shape

# Only keep the first one 
df_timetable = df_timetable.drop_duplicates(['Journey_Pattern_ID','week_cate','Time'])
df_timetable.shape

# Sort the data frame by Journey_Pattern_ID
df_timetable.sort('Journey_Pattern_ID')

# Check columns types 
df_timetable.dtypes

# Check the Journey_Pattern_ID type to string 
df_timetable['Journey_Pattern_ID']=df_timetable['Journey_Pattern_ID'].astype(str)

# Add Line_ID column to the data frame 
df_timetable['Line_ID']= df_timetable['Journey_Pattern_ID'].apply(lambda x:x[1:4:].strip("0"))

# Check the first row of the timetable 
df_timetable.head(1)

# Drop columns which we don't need any more 
df_timetable.drop(['datetime','day_of_week'],axis=1,inplace=True)
df_timetable.drop(['Timestamp'],axis=1,inplace=True)

# Save the timetable to csv file 
df_timetable.to_csv('time_table.csv',index=False)

get_ipython().magic("time df_tt= pd.read_csv('time_table.csv',dtype={ 'Journey_Pattern_ID': object})")
#df_all.dtypes

df_tt.head(1)

# Change the column name for import data to RDS 
df_tt.columns = ['journey_pattern', 'first_stop','last_stop','day_category','departure_time','line']
df_tt.to_csv('time_table.csv',index=False)

df_tt.dtypes

workday_time=[]
for index, row in df_monday.iterrows():
    begin_time=row['Time']
    workday_time.append(begin_time)
print(workday_time)

# Use datetime package to calulate time 
from datetime import datetime
from datetime import timedelta

travel_time=timedelta(hours=0, minutes=30, seconds=0)

begin_time=datetime.strptime('10:35:00','%H:%M:%S')

bus_time=[]
for i in workday_time:
# Parse the time strings
    i = datetime.strptime(i,'%H:%M:%S')
    i+=travel_time
    if i>begin_time:
        bus_time.append(i)
        


print(bus_time)

First_bus=min(bus_time)

#First_bus=First_bus.time()
#print("The first bus will arrive at ",First_bus)

First_bus=First_bus.time()
print("The first bus will arrive at ",First_bus)

