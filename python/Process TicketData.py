import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
import random

# Read in Data
TicketData = pd.read_csv('Data/MasterTicketData.csv')

# Initial Data frame characteristics
num_events = TicketData['event_id'].count()
num_events_w_face_values = TicketData[TicketData['face_value'].str.contains('\$')]['event_id'].count()

print("Total number of events: %s" % num_events)
print("Total number of events with face values: %s \n" % num_events_w_face_values)
print("City counts: \n")
print TicketData['city'].value_counts()

# Get rid of rows where face value is non-numeric
TicketData = TicketData[TicketData['face_value'].str.contains('\$')]

# Remove dollar sign from the face_value field (so that we can compare it to other price fields)
TicketData['face_value'] = TicketData['face_value'].map(lambda x: x.lstrip('$'))

# Convert face_value column to numeric type
TicketData['face_value'] = pd.to_numeric(TicketData['face_value'], errors='ignore')

# Calculate delta between min stubhub price and face value
TicketData['FV_delta'] = TicketData['min_price'] - TicketData['face_value']

# Calculate delta between max stubhub price and face value
TicketData['maxPrice_FV_delta'] = TicketData['max_price'] - TicketData['face_value']

# Convert FV_delta to log form
TicketData['FV_delta_log'] = np.log(TicketData['FV_delta'])

# Get rid of event_id #9478397 which is an outlier (stubhub ticket price is $711,671)
TicketData = TicketData[TicketData['event_id'] != 9478397]

# Display result.
TicketData[['date', 'artist', 'venue', 'city', 'min_price', 'max_price', 'face_value', 
            'FV_delta', 'FV_delta_log', 'maxPrice_FV_delta']].head()

# New Dataframe characteristics
num_events = TicketData['event_id'].count()

print("Total number of events: %s \n" % num_events)

print("Max FV_delta: %s" %TicketData['FV_delta'].max())
print("Max maxPrice_FV_delta: %s" % TicketData['maxPrice_FV_delta'].max())

TicketData['FV_delta_log'].plot.box()

def RemoveOutliersFromDataFrame(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    IQR = q3 - q1
    df.drop(df[df[column_name] > q3+1.5*IQR].index, inplace = True)
    df.drop(df[df[column_name] < q1-1.5*IQR].index, inplace = True)
    return df

# Remove outliers from data
TicketData = RemoveOutliersFromDataFrame(TicketData, 'FV_delta_log')

# Drop negative values 
# Not sure if we want to do this - might be OK to have negative values. I.e., concert demand was not high
# TicketData.drop(TicketData[TicketData['FV_delta'] < 0].index, inplace=True)
TicketData['FV_delta_log'].plot.box()

print TicketData.count()
TicketData.head()

# Get rid of rows where Echonest did not return any data
TicketData = TicketData[TicketData['num_news'] != 'error_5']
# Get rid of rows where num_years_active has a null value
TicketData = TicketData[TicketData['num_years_active'].isnull() == False]
# Get rid of rows where FV_delta_log is null (meaning FV_delta was negative)
TicketData = TicketData[TicketData['FV_delta_log'].isnull() == False]
print TicketData.count()

# Uncomment this in order to save processed dataframe as CSV
TicketData.to_csv(path_or_buf="Data/ProcessedTicketData.csv", index=False)



