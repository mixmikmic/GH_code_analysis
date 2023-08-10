import os
import requests

# Some constants
NEWSROOMDB_URL = os.environ['NEWSROOMDB_URL']

# Utilities for loading data from NewsroomDB

def get_table_url(table_name, base_url=NEWSROOMDB_URL):
    return '{}table/json/{}'.format(base_url, table_name)

def get_table_data(table_name):
    url = get_table_url(table_name)
    
    try:
        r = requests.get(url)
        return r.json()
    except:
        print("Request failed. Probably because the response is huge.  We should fix this.")
        return get_table_data(table_name)

shooting_victims_raw = get_table_data('shootings')
print("Loaded {} shooting victims".format(len(shooting_victims_raw)))

import agate
from datetime import datetime, timedelta

# Load raw data into an Agate table

# Agate tries to parse the date and time automatically. It parses the time incorrectly
# as MM:SS instead of HH:MM. We ultimately need a timestamp, which is easily
# parsed by concatenating the date and time, so disable the initial
# auto-parsing of these fields.
column_types = {
    'Date': agate.Text(),
    'Time': agate.Text(),
}
shooting_victims = agate.Table.from_object(shooting_victims_raw, column_types=column_types)

# Calculate a timestamp from the Date and Time columns

def get_timestamp(row, date_col='Date', time_col='Time'):    
    if not row[date_col] or not row[time_col]:
        return None
    
    try:
        timestamp = datetime.strptime("{} {}".format(row[date_col], row[time_col]), "%Y-%m-%d %H:%M")
    except ValueError:
        timestamp = datetime.strptime("{} {}".format(row[date_col], row[time_col]), "%Y-%m-%d %H:%M:%S")
    
    # HACK: There are some bad dates in the data.  Based on visual inspection,
    # we can fix the dates using a couple of rules
    year = timestamp.year
    if year < 20:
        year += 2000
        new_timestamp = timestamp.replace(year=year)
        print("Bad year date in row with id {}. Changing {} to {}.".format(
            row['_id'], timestamp.strftime("%Y-%m-%d"), new_timestamp.strftime("%Y-%m-%d")))
        timestamp = new_timestamp
    elif year == 216:
        new_timestamp = timestamp.replace(year=2016)
        print("Bad year date in row with id {}. Changing {} to {}.".format(
            row['_id'], timestamp.strftime("%Y-%m-%d"), new_timestamp.strftime("%Y-%m-%d")))
        timestamp = new_timestamp
    
    return timestamp

shooting_victims = shooting_victims.compute([
    ('timestamp', agate.Formula(agate.DateTime(), get_timestamp))
])

shooting_victims = shooting_victims.where(lambda row: row['timestamp'] is not None)

def is_weekend(timestamp):
    """Does the timestamp fall between Friday 3 p.m. and Monday 6 a.m."""
    if not timestamp:
        return False
    
    day_of_week = timestamp.weekday()
    
    if day_of_week > 0 and day_of_week < 4:
        return False
    
    if day_of_week == 4:
        # Friday
        
        # Same day, 3 p.m.
        start = datetime(timestamp.year, timestamp.month, timestamp.day, 15)
        
        return timestamp >= start
    
    if day_of_week == 0:
        # Monday
        
        # Same day, 6 a.m.
        end = datetime(timestamp.year, timestamp.month, timestamp.day, 6)
        
        return timestamp < end
        
    return True

weekend_shootings = shooting_victims.where(lambda row: is_weekend(row['timestamp']))
print("There are {0} weekend shooting victims".format(len(weekend_shootings.rows)))

from datetime import datetime
import time

# Utility functions for calculating weekend start and end dates/times for a given 

def clone_datetime(d):
    """Make a copy of a datetime object"""
    # HACK: Is there a better way to do this?  Why isn't there an obvious clone method?
    return datetime.fromtimestamp(time.mktime(d.timetuple()))

# The following methods only work for timestamps that fall within a weekend

def weekend_start(timestamp):
    days_from_friday = timestamp.weekday() - 4
    
    if days_from_friday < 0:
        days_from_friday += 1
        days_from_friday *= -1
        
    friday_delta = timedelta(days=(-1 * days_from_friday))
    
    start = clone_datetime(timestamp)
    
    start += friday_delta
    start = start.replace(hour=15, minute=0, second=0)
    
    return start

def weekend_end(timestamp):
    days_to_monday = 0 - timestamp.weekday()
    
    if days_to_monday < 0:
        days_to_monday += 7
        
    monday_delta = timedelta(days=days_to_monday)
    
    end = clone_datetime(timestamp)
    
    end += monday_delta
    end = end.replace(hour=6, minute=0, second=0)
    
    return end

def get_weekend_start(row):
    return weekend_start(row['timestamp']).date()

# Add weekend start and end dates to each row so we can
# group by on them later.  Cecilia took a different approach,
# calculating the weekends first and iterating through them
# and finding matching shootings for each weekend.
weekend_shootings_with_start_end = weekend_shootings.compute([
    ('weekend_start', agate.Formula(agate.Date(), get_weekend_start)),
    ('weekend_end', agate.Formula(agate.Date(), lambda row: weekend_end(row['timestamp']).date()))
])

# Aggregate the shooting victims by weekend
shooting_victims_by_weekend = weekend_shootings_with_start_end.group_by(
    lambda row: row['weekend_start'].strftime("%Y-%m-%d") + " to " +  row['weekend_end'].strftime("%Y-%m-%d"))

shooting_victims_weekend_counts = shooting_victims_by_weekend.aggregate([
    ('count', agate.Count())
])

shooting_victims_weekend_counts.order_by('count', reverse=True).print_table(max_column_width=40, max_rows=None)

homicides_raw = get_table_data('homicides')

homicide_column_types = {
    'Occ Date': agate.Text(),
    'Occ Time': agate.Text(),
}
homicides = agate.Table.from_object(homicides_raw, column_types=homicide_column_types)
homicides = homicides.compute([
    ('timestamp', agate.Formula(agate.DateTime(), lambda row: get_timestamp(row, date_col='Occ Date', time_col='Occ Time')))
])

weekend_homicides = homicides.where(lambda row: is_weekend(row['timestamp']))
weekend_homicides_with_start_end = weekend_homicides.compute([
    ('weekend_start', agate.Formula(agate.Date(), get_weekend_start)),
    ('weekend_end', agate.Formula(agate.Date(), lambda row: weekend_end(row['timestamp']).date()))
])

homicides_by_weekend = weekend_homicides_with_start_end.group_by(
    lambda row: row['weekend_start'].strftime("%Y-%m-%d") + " to " +  row['weekend_end'].strftime("%Y-%m-%d"))

weekend_homicide_counts = homicides_by_weekend.aggregate([
    ('count', agate.Count())
])

weekend_homicide_counts.order_by('count', reverse=True).print_table(max_column_width=40, max_rows=None)

import re

# First off, we need to avoid double-counting homicides and shootings
def is_homicide(row):
    if not row['UCR']:
        return False
    
    if re.match(r'0{0,1}110', row['UCR']):
        return True
    
    return False

non_homicide_weekend_shootings = weekend_shootings_with_start_end.where(lambda row: not is_homicide(row))
print("There are {0} non-homicide weekend shootings".format(len(non_homicide_weekend_shootings.rows)))

non_homicide_shooting_victims_by_weekend = non_homicide_weekend_shootings.group_by(
    lambda row: row['weekend_start'].strftime("%Y-%m-%d") + " to " +  row['weekend_end'].strftime("%Y-%m-%d"))

non_homicide_shooting_victims_weekend_counts = non_homicide_shooting_victims_by_weekend.aggregate([
    ('count', agate.Count())
])

def none_to_zero(x):
    if x is None:
        return 0
    
    return x

shooting_victims_and_homicides = non_homicide_shooting_victims_weekend_counts.join(weekend_homicide_counts, 'group')
shooting_victims_and_homicides = shooting_victims_and_homicides.compute([
    ('total', agate.Formula(agate.Number(), lambda row: row['count'] + none_to_zero(row['count2']))),
])
shooting_victims_and_homicides.order_by('total', reverse=True).print_table(max_column_width=40, max_rows=None)



