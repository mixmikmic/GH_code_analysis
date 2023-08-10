import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import time
get_ipython().magic('matplotlib inline')

main_data = pd.read_csv("data_with_station_cities.csv", delimiter=",", low_memory=False)

print "Dimension of the data: ", main_data.shape
main_data.head()

# number of trips
n =  main_data.shape[0]

# extract the time columns
start_time = main_data['starttime']
stop_time = main_data['stoptime']
# store the results
month = []
day_of_month = []
day_of_week = []
start_hour = []
stop_hour = []

# for each trip
for i in range(n):
    # separate start time into month, day, start_hour
    trip_start_time = time.strptime(start_time[i], "%m/%d/%Y %H:%M:%S")
    month.append(trip_start_time.tm_mon)
    day_of_month.append(trip_start_time.tm_mday)
    start_hour.append(trip_start_time.tm_hour)
    # extract weekday
    day_of_week.append(trip_start_time.tm_wday)

    # extract stop_hour from stop_time
    trip_stop_time = time.strptime(stop_time[i], "%m/%d/%Y %H:%M:%S")
    stop_hour.append(trip_stop_time.tm_hour)

# organize useful information to a new dataframe
new_data = pd.DataFrame({
        "month": month,
        "day_of_month": day_of_month,
        "day_of_week": day_of_week,
        "start_hour": start_hour,
        "stop_hour": stop_hour,        
    })

new_data = pd.concat([new_data, main_data['tripduration'], main_data['start_neighborhood'], 
                      main_data['end_neighborhood'], main_data['usertype']], axis = 1)

new_data.head()

new_data.to_csv("new_trip_data.csv", sep = ',')

data_organized = []
# create labels
labels = ["month", "day_of_month", "day_of_week", "neighborhood", "hour", "trip_number", 
          "average_duration", "percent_subscriber", "trip_type"]
data_organized = np.asarray(labels).reshape(1, 9)

for month_num in new_data['month'].unique():
    # operate on data of this month
    month_data = new_data.loc[new_data['month'] == month_num]
    for day_num in month_data['day_of_month'].unique():
        # operate on data of each day
        day_data = month_data.loc[month_data['day_of_month'] == day_num]
        weekday = day_data['day_of_week'].values[0]
        # outgoing trips in each region
        for region in day_data['start_neighborhood'].unique():
            region_data = day_data.loc[day_data['start_neighborhood'] == region]
            # compute for each hour
            for hour in region_data['start_hour'].unique():
                # prepare information of this piece of data
                summary = []
                summary.append(month_num)
                summary.append(day_num)
                summary.append(weekday)
                summary.append(region)
                summary.append(hour)
                # target dataset
                final_data = region_data.loc[region_data['start_hour'] == hour]
                # compute number of outgoing trips
                summary.append(final_data.shape[0])
                # average trip duration
                summary.append(np.mean(final_data['tripduration'].values))
                # compute subscriber percentage
                percent = np.mean(final_data['usertype'] == "Subscriber")
                summary.append(percent)
                # label trip type
                summary.append("outgoing")
                
                # append the list to a large array
                summary = np.asarray(summary).reshape((1, 9))
                data_organized = np.concatenate((data_organized, summary), axis = 0)

        # incoming trips in each region   
        for region in day_data['end_neighborhood'].unique():
            region_data = day_data.loc[day_data['end_neighborhood'] == region]
            # compute for each hour
            for hour in region_data['stop_hour'].unique():
                # prepare information of this piece of data
                summary = []
                summary.append(month_num)
                summary.append(day_num)
                summary.append(weekday)
                summary.append(region)
                summary.append(hour)
                # target dataset
                final_data = region_data.loc[region_data['stop_hour'] == hour]
                # compute number of incoming trips
                summary.append(final_data.shape[0])
                # average trip duration
                summary.append(np.mean(final_data['tripduration'].values))
                # compute subscriber percentage
                percent = np.mean(final_data['usertype'] == "Subscriber")
                summary.append(percent)
                # label trip type
                summary.append("incoming")
                
                # append the list to a large array
                summary = np.asarray(summary).reshape((1, 9))
                data_organized = np.concatenate((data_organized, summary), axis = 0)

print "Dimension: ", data_organized.shape 

df_data_organized = pd.DataFrame(data_organized[1:, :], columns= data_organized[0, :])
df_data_organized.head()

station_data = pd.read_csv("station_info.csv", delimiter=",", low_memory=False)
print "Dimension of station data: ", station_data.shape
station_data.head()

station_in_service = station_data.loc[station_data['statusKey'] == 1]
print "Dimension of station in-service: ", station_in_service.shape

# match the stations to neighborhood
n_station =  station_in_service.shape[0]
station_neighborhood = []

# 0 if using the start_neighborhood, 1 if using the end_neighborhood
end_station = 0

for n in range(n_station):
    station_id = station_in_service['id'].values[n]
    same_station = main_data.loc[main_data['start.station.id'] == station_id]
    if same_station.shape[0] == 0:
        same_station = main_data.loc[main_data['end.station.id'] == station_id]
        end_station = 1
    if same_station.shape[0] == 0:
        region = 'NA'
    elif end_station == 0:
        region = same_station['start_neighborhood'].values[0]
    else:
        region = same_station['end_neighborhood'].values[0]
    end_station = 0
    station_neighborhood.append(region)
    
print "Check dimension of the matched neighborhood: ", len(station_neighborhood)

station_neighborhood = np.asarray(station_neighborhood)

station_not_in_service = station_neighborhood[station_neighborhood == "NA"]

print "Number of stations not-in-service: ", len(station_not_in_service)

station_in_service['id'].values[station_neighborhood == "NA"]

station_in_service['neighborhood'] = pd.Series(station_neighborhood)

# check how many regions there are
station_in_service['neighborhood'].unique()

# compute total docks in each region
docks_by_region = ["neighborbood", "totalDocks"]
docks_by_region = np.array(docks_by_region).reshape((1,2))

for region in station_in_service['neighborhood'].unique():
    region_data = station_in_service.loc[station_in_service['neighborhood'] == region]
    result = [region, np.sum(region_data['totalDocks'].values)]
    result = np.asarray(result).reshape((1, 2))
    
    docks_by_region = np.concatenate((docks_by_region, result), axis = 0)

docks_by_region

# normalize the trip numbers by total docks in the region
n_regions = docks_by_region.shape[0] - 1

# extract information from the organized data
neighborhood = data_organized[1:,3]
trip_number = data_organized[1:,5].astype(int)
# array to store the normalized trip number
normalized_trip_num = np.zeros(len(trip_number))

for n in range(n_regions):
    region = docks_by_region[n+1, 0]
    normalized_trip_num[neighborhood == region] = trip_number[neighborhood == region]/float(docks_by_region[n+1, 1])

# add to the dataframe
df_data_organized['normalized_trip_num'] = normalized_trip_num

df_data_organized.head()

# function to convert hours into groups
# input: hours 0-24
# output: new group

def match(input_hour):
    if input_hour < 6 or input_hour > 21:
        output_hour = 'night'
    elif input_hour < 8:
        output_hour = '7'
    elif input_hour < 9:
        output_hour = '8'
    elif input_hour < 10:
        output_hour = '9'
    elif input_hour < 17:
        output_hour = '10-16'
    elif input_hour < 18:
        output_hour = '17'
    elif input_hour < 19:
        output_hour = '18'
    elif input_hour < 20:
        output_hour = '19'
    elif input_hour < 21:
        output_hour = '20'
    else:
        output_hour = '21'
    
    return (output_hour)

hours = df_data_organized['hour'].values.astype(int)
hour_group = []

for n in range(len(hours)):
    group = match(hours[n])
    hour_group.append(group)
    
hour_group = np.asarray(hour_group).reshape((len(hours), 1))

# combined with the organized data
df_data_organized['hour_group'] = hour_group

# save the data
df_data_organized.to_csv("organized_data.csv", sep = ',')

data_organized_2 = []
# create labels
labels = ["month", "day_of_month", "day_of_week", "neighborhood", "hour", "normalized_imbalance"]
data_organized_2 = np.asarray(labels).reshape(1, 6)

for month_num in df_data_organized['month'].unique():
    # operate on data of this month
    month_data = df_data_organized.loc[df_data_organized['month'] == month_num]
    for day_num in month_data['day_of_month'].unique():
        # operate on data of each day
        day_data = month_data.loc[month_data['day_of_month'] == day_num]
        weekday = day_data['day_of_week'].values[0]
        # for each region
        for region in day_data['neighborhood'].unique():
            region_data = day_data.loc[day_data['neighborhood'] == region]
            # compute for each hour
            for hour in region_data['hour'].unique():
                # prepare information of this piece of data
                summary = []
                summary.append(month_num)
                summary.append(day_num)
                summary.append(weekday)
                summary.append(region)
                summary.append(hour)
                # target dataset
                final_data = region_data.loc[region_data['hour'] == hour]
                
                # compute normalized imbalance
                outgoing_data = final_data.loc[final_data['trip_type'] == 'outgoing']['normalized_trip_num'].values
                if len(outgoing_data) > 0:
                    outgoing = outgoing_data[0]
                else:
                    # set to 0 if no outgoing trips
                    outgoing = 0
                    
                incoming_data = final_data.loc[final_data['trip_type'] == 'incoming']['normalized_trip_num'].values
                if len(incoming_data) > 0:
                    incoming = incoming_data[0]
                else:
                    # set to 0 if no incoming trips
                    incoming = 0
                
                imbalance = outgoing - incoming
                summary.append(imbalance)
                
                # append the list to a large array
                summary = np.asarray(summary).reshape((1, 6))
                data_organized_2 = np.concatenate((data_organized_2, summary), axis = 0)      

# convert to dataframe
df_data_organized_2 = pd.DataFrame(data_organized_2[1:, :], columns= data_organized_2[0, :])

# add hour group
hours = df_data_organized_2['hour'].values.astype(int)
hour_group = []

for n in range(len(hours)):
    group = match(hours[n])
    hour_group.append(group)
    
hour_group = np.asarray(hour_group).reshape((len(hours), 1))

# combined with the organized data
df_data_organized_2['hour_group'] = hour_group

df_data_organized_2.head()

# save the data
df_data_organized_2.to_csv("organized_data_2.csv", sep = ',')

data_2 = pd.read_csv("organized_data.csv", delimiter=",", low_memory=False)
data_2.head()

# function to compute total trips, % subscriber and average trip duration
# given a dataframe of the date, region and triptype
# return the summary list with new data appended
def data_summary (dataset, hour1, hour2, summary):
    outgoing_1 = dataset.loc[dataset['hour'] == hour1]
    outgoing_2 = dataset.loc[dataset['hour'] == hour2]
    # trip numbers
    total_trips = outgoing_1['trip_number'].values + outgoing_2['trip_number'].values
    if total_trips.shape[0] == 0:
        summary.append(0)
        summary.append(0)
        summary.append(0)
    else:           
        total_trips = total_trips[0]
        summary.append(total_trips)
        # %subscriber
        usertype = (outgoing_1['percent_subscriber'].values * outgoing_1['trip_number'].values 
        + outgoing_2['percent_subscriber'].values * outgoing_2['trip_number'].values)/total_trips
        summary.append(usertype[0])
        # average trip duration
        duration = (outgoing_1['average_duration'].values * outgoing_1['trip_number'].values 
        + outgoing_2['average_duration'].values * outgoing_2['trip_number'].values)/total_trips
        summary.append(duration[0])
    
    return summary

data_organized_3 = []
# create labels
labels = ["month", "day_of_month", "day_of_week", "neighborhood", "morning_outgoing", "monring_out_user", "morning_out_duration",
          "afternoon_outgoing", "afternoon_out_user", "afternoon_out_duration", "morning_incoming", "monring_in_user", 
          "morning_in_duration", "afternoon_incoming", "afternoon_in_user", "afternoon_in_duration"]
data_organized_3 = np.asarray(labels).reshape(1, 16)


for month_num in data_2['month'].unique():
    # operate on data of this month
    month_data = data_2[data_2['month'] == month_num]
    for day_num in month_data['day_of_month'].unique():
        # operate on data of each day
        day_data = month_data.loc[month_data['day_of_month'] == day_num]
        weekday = day_data['day_of_week'].values[0]
        # for each region
        for region in day_data['neighborhood'].unique():
            region_data = day_data.loc[day_data['neighborhood'] == region]
            # prepare information of this piece of data
            summary = []
            summary.append(month_num)
            summary.append(day_num)
            summary.append(weekday)
            summary.append(region)
            
            # outgoing
            outgoing_data = region_data.loc[region_data['trip_type'] == 'outgoing']
            # morning rush hour
            summary = data_summary (outgoing_data, 8, 9, summary)
            # afternoon rush hour
            summary = data_summary (outgoing_data, 17, 18, summary)
            
            # incoming
            incoming_data = region_data.loc[region_data['trip_type'] == 'incoming']
            # morning rush hour
            summary = data_summary (incoming_data, 8, 9, summary)
            # afternoon rush hour
            summary = data_summary (incoming_data, 17, 18, summary)
            
            # append the list to a large array
            summary = np.asarray(summary).reshape((1, 16))
            data_organized_3 = np.concatenate((data_organized_3, summary), axis = 0)      

# convert to dataframe
df_data_organized_3 = pd.DataFrame(data_organized_3[1:, :], columns= data_organized_3[0, :])
df_data_organized_3.head()

# save the data
df_data_organized_3.to_csv("organized_data_3.csv", sep = ',')



