import datetime
import pandas as pd
import numpy as np
import psycopg2
import csv
import time
from datetime import date

try:
    connection = psycopg2.connect(database ='heatseek', user = 'heatseekroot', password = 'wearecoolsoweseekheat')
    cursor = connection.cursor() #Open a cursor to perform operations
    
    cursor.execute('SELECT * from users') #Executes the query
    users = cursor.fetchall() #cursor.fetchone() for one line, fetchmany() for multiple lines, fetchall() for all lines
    users = pd.DataFrame(users) #Saves 'users' as a pandas dataframe
    users_header = [desc[0] for desc in cursor.description] #This gets the descriptions from cursor.description 
    #(names are in the 0th index)
    users.columns = users_header #PD array's column names
    
    cursor.execute('SELECT * FROM readings;')
    readings = cursor.fetchall()
    readings = pd.DataFrame(readings)
    readings_header = [desc[0] for desc in cursor.description]
    readings.columns = readings_header
    
    cursor.execute('SELECT * FROM sensors;')
    sensors = cursor.fetchall()
    sensors = pd.DataFrame(sensors)
    sensors_header = [desc[0] for desc in cursor.description]
    sensors.columns = sensors_header
    
    cursor.close() 
    connection.close()
    
except psycopg2.DatabaseError, error:
    print 'Error %s' % error

#This creates an array 'sensors_with_users' that consists of sensors that are currently assigned to users.
sensors_with_users = sorted([x for x in users.id.unique() if x in  sensors.user_id.unique()])

#This function returns clean readings. #It doesn't exist yet
#This function will return if a sensor is polling faster than once per hour (i.e., test cases)

def dirty_data(dirty_readings, start_date = None, end_date = None):
    if (start_date or end_date) == None:
        start_date = pd.Timestamp('2000-01-01')
        end_date = pd.Timestamp(datetime.datetime.now())
    else:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    
    mask = (dirty_readings['created_at'] > start_date) & (dirty_readings['created_at'] <= end_date)
    dirty_readings = dirty_readings.loc[mask]
    
    hot_ids = dirty_readings.loc[dirty_readings.temp > 90].sensor_id.unique() #Returns sensor IDs where indoor temp is > 90
    cold_ids = dirty_readings.loc[dirty_readings.temp < 40].sensor_id.unique() #Returns sensor IDs where indoor temp is < 40
    inside_colder_ids = dirty_readings.loc[dirty_readings.temp < dirty_readings.outdoor_temp].sensor_id.unique() #Returns sensor IDs where indoor temp is < outdoor temp
    #Array of all the IDs above
    all_ids = np.unique(np.concatenate((hot_ids, cold_ids, inside_colder_ids)))
    all_ids = all_ids[~np.isnan(all_ids)]
    #Create an empty dataframe with the IDs as indices
    report = pd.DataFrame(index=all_ids,columns=['UserID','SensorID', 'Outside90', 'Inside40', 'InsideColderOutside'])
    #Fill in the specific conditions as '1'
    report.Outside90 = report.loc[hot_ids].Outside90.fillna(1)
    report.Inside40 = report.loc[cold_ids].Inside40.fillna(1)
    report.InsideColderOutside = report.loc[inside_colder_ids].InsideColderOutside.fillna(1)
    report = report.fillna(0)
    report.SensorID = report.index
    
    #Fill in UserIDs
    problem_ids = sensors[sensors.id.isin(all_ids)]
    for index in report.index.values:
        index = int(index)
        try:
            report.loc[index, 'UserID'] = sensors.loc[index, 'user_id']
        except KeyError:
            report.loc[index, 'UserID']  = 'No such user in sensors table.'
    return report

def clean_data(dirty_readings):
    cleaner_readings = dirty_readings[dirty_readings.sensor_id.notnull()] #Remove cases where there are no sensor IDs
    return cleaner_readings
    

#This function takes (start date, end date, sensor id), returns % of failure
def sensor_down(data, start_date, end_date, sensor_id): 
    
    #This pulls up the tennant's first and last name.
    try:
        tennant_id = int(sensors.loc[sensors.id == sensor_id].user_id.values[0])
        tennant_first_name = users.loc[users.id == tennant_id].first_name.values[-1] #This pulls up the first name on the list (not the most recent)
        tennant_last_name = users.loc[users.id == tennant_id].last_name.values[-1]
    #Are these really not assigned?
    except ValueError:
        tennant_id = 'None'
        tennant_first_name = 'Not'
        tennant_last_name = 'Assigned'
    except IndexError:
        tennant_id = 'None'
        tennant_first_name = 'Not'
        tennant_last_name = 'Assigned'
        
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    sensor_readings = data.loc[data.sensor_id == sensor_id]
    
    #Converting to timestamps
    #for i in sensor_readings.index.values: #Iterates through all the index values
        #sensor_readings.loc[i, 'created_at'] = pd.Timestamp(sensor_readings.created_at[i])
    #Using map instead of for loop (about 15-20x faster)
    try:
        sensor_readings.loc[:, 'created_at'] = map(pd.Timestamp, sensor_readings.created_at)
    except TypeError:
        tennant_first_name = 'Mapping Error'
        tennant_last_name = 'Only One Entry'
        pass
    #Using list comprehensions (as efficient as map)
    #sensor_readings.loc[:, 'created_at'] = [pd.Timestamp(x) for x in sensor_readings.created_at]
        
    #Using a boolean mask to select readings between the two dates 
    #(http://stackoverflow.com/questions/29370057/select-dataframe-rows-between-two-dates)   
    mask = (sensor_readings['created_at'] > start_date) & (sensor_readings['created_at'] <= end_date)
    masked_sensor_readings = sensor_readings.loc[mask] #Get all readings between the two dates
    masked_sensor_readings = masked_sensor_readings.sort_values('created_at')
    #We then calculate how many hours have passed for that specific sensor and date range
    try:
        sensor_readings_start_date = masked_sensor_readings.loc[masked_sensor_readings.index.values[0], 'created_at']
        sensor_readings_end_date =         masked_sensor_readings.loc[masked_sensor_readings.index.values[len(masked_sensor_readings)-1], 'created_at']
        timedelta_in_seconds =  sensor_readings_end_date - sensor_readings_start_date #This returns Timedelta object
        timedelta_in_seconds = timedelta_in_seconds.total_seconds()
        total_number_of_hours = timedelta_in_seconds/3600 + 1 #The +1 fixes the rounding error for now but IDK why yet.
        
        hours_in_date_range = ((end_date-start_date).total_seconds())/3600 + 1
        
    except IndexError:
        return [tennant_first_name, tennant_last_name, sensor_id, tennant_id, "No valid readings during this time frame."]
    
    proportion_of_total_uptime = (len(masked_sensor_readings)/hours_in_date_range) * 100 #Proportion of uptime over TOTAL HOURS
    proportion_within_sensor_uptime = (len(masked_sensor_readings)/total_number_of_hours) * 100 #Proportion of uptime for the sensor's first and last uploaded dates.
    if proportion_within_sensor_uptime <= 100.1:
        return [tennant_first_name, tennant_last_name, sensor_id, tennant_id, proportion_of_total_uptime, proportion_within_sensor_uptime]
    else:
        return [tennant_first_name, tennant_last_name, sensor_id, tennant_id, proportion_of_total_uptime, proportion_within_sensor_uptime, 'Sensor has readings more frequent than once per hour. Check readings table.']

def violation_percentages(data, start_date, end_date, sensor_id):
    
    sensor_readings = data.loc[data.sensor_id == sensor_id] #All readings for a sensorID
    try:
        sensor_readings.loc[:,'created_at'] = map(pd.Timestamp, sensor_readings.created_at) #convert all to timestampst
    except TypeError:
        pass
    
    #Filter out sensors that are < 30 days old
    try:
        sensor_readings_start_date = sensor_readings.loc[sensor_readings.index.values[0], 'created_at'].date()
        today = date.today()
        datediff = today - sensor_readings_start_date
    except:
        return "No readings in date range."
    
    if datediff.days < 30: #If a sensor has been up for < 30 days, don't do anything
        pass
    else:
        start_date = pd.Timestamp(start_date) #Convert dates to pd.Timestamp
        end_date = pd.Timestamp(end_date)
    
        mask = (sensor_readings['created_at'] > start_date) & (sensor_readings['created_at'] <= end_date) #mask for date range
        masked_sensor_readings = sensor_readings.loc[mask]

        try:
            #First, find all possible violation-hours
            ##We need to index as datetimeindex in order to use the .between_time method
            sensor_readings.set_index(pd.DatetimeIndex(sensor_readings['created_at']), inplace = True)
        
            ##This returns the a list of day and night readings
            day_readings = sensor_readings.between_time(start_time='06:00', end_time='22:00')
            night_readings = sensor_readings.between_time(start_time='22:00', end_time='6:00')

            ##Now, we count how many rows are violations and divide by total possible violation hours
            #For day, if outdoor_temp < 55
            day_total_violable_hours = len(day_readings.loc[day_readings['outdoor_temp'] < 55])
            day_actual_violation_hours = len(day_readings.loc[day_readings['violation'] == True])
            #For night, if outdoor_temp < 40
            night_total_violable_hours = len(night_readings.loc[night_readings['outdoor_temp'] < 40])
            night_actual_violation_hours = len(night_readings.loc[night_readings['violation'] == True])

            #Calculate percentage
            try:
                violation_percentage = float(day_actual_violation_hours + night_actual_violation_hours)/float(day_total_violable_hours + night_total_violable_hours)
            except ZeroDivisionError:
                return "No violations in this range."
                
            return violation_percentage #violationpercentage
        
        except IndexError:
            pass

def violation_report():
    unique_sensors = readings['sensor_id'].unique()
    report = []
    for ids in unique_sensors:
         report.append("Sensor ID: {0}, Violation Percentage: {1}".format(ids, violation_percentages(readings, '2016-01-01', '2016-02-07', ids)))
    return report

#This function creates a simulated dataset of readings.
def simulate_data(start_date, end_date, polling_rate, sensor_id): #polling_rate in minutes
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    #how many hours between the two dates:
    timedelta_in_seconds = end_date-start_date
    total_number_of_hours = timedelta_in_seconds.total_seconds()/(polling_rate*60)
    
    #Create an empty pandas dataframe
    index = xrange(1,int(total_number_of_hours)+1)
    columns = ['created_at', 'sensor_id']
    simulated_readings = pd.DataFrame(index = index, columns = columns)
    simulated_readings.loc[:,'sensor_id'] = sensor_id
    
    #Populate it with columns of 'create_at' dates
    time_counter = start_date
    for i in simulated_readings.index.values:
        simulated_readings.loc[i,'created_at'] = time_counter
        time_counter = time_counter + pd.Timedelta('00:%s:00' % polling_rate)
   
    return simulated_readings

#This function generates a report; we might want to make this a cron job.
def generate_report(start_date, end_date):
    report = []
    sensor_ids = readings.sensor_id.unique()
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    for ids in sensor_ids:
        temp = sensor_down(readings, start_date, end_date, ids)
        if temp != None:
            report.append(temp)
        else:
            pass
    return report

tic = time.clock()

report = generate_report('2016-02-01','2016-02-07')
header =['sensorID', 'status', 'Percentage of uptime in daterange', 'FirstName', 'LastName' , 'userID']

toc = time.clock()
toc - tic

tic = time.clock()

report = dirty_data(readings)
header = ['UserID', 'SensorID', 'Outside90', 'Inside40', 'InsideColderOutside']

toc = time.clock()
toc - tic

csvoutput = open('sensors.csv', 'wb')
writer = csv.writer(csvoutput)
writer.writerow(header)
for i in report:
    writer.writerow(i)
csvoutput.close()

report.to_csv('dirtydata.csv', index = False, na_rep="Not Currently Assigned")

