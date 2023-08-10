import requests
import json
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import re
import datetime
from datetime import timedelta
from dateutil import relativedelta
import calendar
import getpass
import psycopg2
from sqlalchemy import create_engine

host = raw_input("Host Address: ")

db_name = "pokemon_go"
username = "pokemon_go_role"

password = getpass.getpass()

years = xrange(2016, 2025)

start_date = str(datetime.date(years[0], 1, 1))
end_date = str(datetime.date(years[-1], 12, 31))

all_dates = pd.date_range(start_date, end_date, freq='D')
all_dates

# Given a datetime timestamp, produce a datekey
def datetimeToDateKey(date):
    date_str = str(date)
    date_str = date_str.split(" ")[0]
    date_str = re.sub('-', '', date_str)
    return date_str

# Given a datetime timestamp, convert it to a string of just the date in YYYY-mm-dd format
def fullDate(date):
    date_str = str(date)
    date_str = date_str.split(" ")[0]
    return date_str

# Returns the weekday as a number
def weekdayNum(date):
    return date.isoweekday()

# Returns the weekday as a string in its full-length form
def weekdayStr(date):
    return date.strftime("%A")

# Returns the weekday as a string in its abbreviated form
def weekdayStrAbv(date):
    return date.strftime("%a")

# Returns the day of the month
def dayNumMonth(date):
    return date.day

# Numbers each day, constantly increasing from the first day
def dayNumOverall(date):
    day_one = all_dates[0]
    date_diff = date - day_one
    return date_diff.days + 1

# Returns the day of the month
def isWeekday(date):
    if date.isoweekday() in range(1,6):
        return "Weekday"
    else:
        return "Weekend"

# Returns the day of the month
def weekNum(date):
    return date.isocalendar()[1]

def weekBeginDate(date):
    dow = date.isoweekday()
    week_start = date - timedelta(days=(dow - 1))
    return week_start

def weekBeginDateKey(date):
    week_begin_date = weekBeginDate(date)
    date_key = datetimeToDateKey(week_begin_date)
    return int(date_key)

# Numbers each day, constantly increasing from the first day
def weekNumOverall(date):
    # Find the date that the first week in the entire data set starts
    first_day = all_dates[0]
    first_week_start = weekBeginDate(first_day)
    
    # Find the date that starts the week of the current date
    curr_week_start = weekBeginDate(date)
    
    # Get the difference and find out how many weeks have passed
    date_diff = curr_week_start - first_week_start
    week_number = int(date_diff.days / 7.0 + 1.0)
    return week_number

# Returns the weekday as a number
def monthNum(date):
    return date.month

# Returns the weekday as a number
def monthNumOverall(date):
    start_date = all_dates[0]
    rel_date = relativedelta.relativedelta(date, start_date)
    month_diff = rel_date.years * 12 + rel_date.months
    return month_diff + 1

# Returns the month as a string in its full-length form
def monthStr(date):
    return date.strftime("%B")

# Returns the month as a string in its abbreviated form
def monthStrAbv(date):
    return date.strftime("%b")

# Returns the quarter in the year
def quarter(date):
    month = date.month
    quarter = month / 4 + 1
    return quarter

# Returns the year as a string
def year(date):
    return date.strftime("%Y")

# Returns the year and month as a concatenated string
def yearmo(date):
    year = date.strftime("%Y")
    month = date.strftime("%m")
    return year + month

# Returns whether or not the date is the last day of the month
def isMonthEnd(date):
    year = date.year
    month = date.month
    
    month_end = calendar.monthrange(year, month)[1]
    
    if (month_end == date.day):
        return "Month End"
    else:
        return "Not Month End"

# Use the date functions to make a dateframe

# Dates
date_dim = DataFrame(all_dates, columns=["full_date"])
date_dim["date_key"] = date_dim["full_date"].map(datetimeToDateKey)
date_dim = date_dim[['date_key', 'full_date']] # Reorder

# Days of Week
date_dim["day_of_week"] = date_dim["full_date"].map(weekdayNum)
date_dim["day_of_week_name"] = date_dim["full_date"].map(weekdayStr)
date_dim["day_of_week_name_abbrev"] = date_dim["full_date"].map(weekdayStrAbv)

date_dim["day_of_month"] = date_dim["full_date"].map(dayNumMonth)
date_dim["day_number_overall"] = date_dim["full_date"].map(dayNumOverall)
date_dim["day_number_overall"] = date_dim["full_date"].map(dayNumOverall)

date_dim["weekday_flag"] = date_dim["full_date"].map(isWeekday)
date_dim["week_number"] = date_dim["full_date"].map(weekNum)
date_dim["week_number_overall"] = date_dim["full_date"].map(weekNumOverall)

date_dim["week_begin_date"] = date_dim["full_date"].map(weekBeginDate)
date_dim["week_begin_date_key"] = date_dim["full_date"].map(weekBeginDateKey)

date_dim["month_number"] = date_dim["full_date"].map(monthNum)
date_dim["month_number_overall"] = date_dim["full_date"].map(monthNumOverall)
date_dim["month"] = date_dim["full_date"].map(monthStr)
date_dim["month_abbrev"] = date_dim["full_date"].map(monthStrAbv)

date_dim["quarter"] = date_dim["full_date"].map(quarter)

date_dim["year"] = date_dim["full_date"].map(year)
date_dim["year_month"] = date_dim["full_date"].map(yearmo)

date_dim["month_end_flag"] = date_dim["full_date"].map(isMonthEnd)

from datetime import time
import math

# 1440 minutes in a day
minutes = xrange(0,1440)

# Given a minute number, return the 12-hour time label
def time_label_12(min_num):
    hours, minutes = divmod(int(min_num), 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%I:%M %p'))

# Given a minute number, return the 24-hour time label
def time_label_24(min_num):
    hours, minutes = divmod(int(min_num), 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%H:%M'))

# Given a minute number, return the 15 minute interval it occures in
def time_interval_15_min(min_num):
    return str(int(math.floor(min_num / 15.0)))

# Given a minute number, return the 30 minute interval it occures in
def time_interval_30_min(min_num):
    return str(int(math.floor(min_num / 30.0)))

# Given a minute number, return the 60 minute interval it occures in
def time_interval_60_min(min_num):
    return str(int(math.floor(min_num / 60.0)))

# Given a minute number, return the 12-hour time label 
# with only hours (this takes up less space and is useful in some cases)
def label_hh(min_num):
    hours, minutes = divmod(int(min_num), 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%I %p'))

# Given a minute number, return the 24-hour time label with just hours
def label_hh24(min_num):
    hours, minutes = divmod(int(min_num), 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%H'))

# Given a minute number, return the 15 minute interval label for a 24-hour clock
def label_15_min_24(min_num):
    interval_num = time_interval_15_min(min_num)
    int_min_num = int(interval_num) * 15
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%H:%M'))

# Given a minute number, return the 30 minute interval label for a 24-hour clock
def label_30_min_24(min_num):
    interval_num = time_interval_30_min(min_num)
    int_min_num = int(interval_num) * 30
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%H:%M'))

# Given a minute number, return the 60 minute interval label for a 24-hour clock
def label_60_min_24(min_num):
    interval_num = time_interval_60_min(min_num)
    int_min_num = int(interval_num) * 60
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%H:%M'))

# Given a minute number, return the 15 minute interval label for a 12-hour clock
def label_15_min_12(min_num):
    interval_num = time_interval_15_min(min_num)
    int_min_num = int(interval_num) * 15
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%I:%M %p'))

# Given a minute number, return the 30 minute interval label for a 12-hour clock
def label_30_min_12(min_num):
    interval_num = time_interval_30_min(min_num)
    int_min_num = int(interval_num) * 30
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%I:%M %p'))

# Given a miute number, return the 60 minute interval label for a 12-hour clock
def label_60_min_12(min_num):
    interval_num = time_interval_60_min(min_num)
    int_min_num = int(interval_num) * 60
    hours, minutes = divmod(int_min_num, 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(time.strftime(timestamp, '%I:%M %p'))

# Given a minute, return just the minute portion
def minute_after_hour(min_num):
    hours, minutes = divmod(int(min_num), 60)
    timestamp = time(hour=hours, minute=minutes)
    return str(int(time.strftime(timestamp, '%M')))

minute_after_hour(61)

time_dim = DataFrame(Series(minutes), columns=["time_key"])
time_dim["time_label_24"] = time_dim["time_key"].map(time_label_24)
time_dim["time_label_12"] = time_dim["time_key"].map(time_label_12)

time_dim["time_interval_15min"] = time_dim["time_key"].map(time_interval_15_min)
time_dim["time_interval_30min"] = time_dim["time_key"].map(time_interval_30_min)
time_dim["time_interval_60min"] = time_dim["time_key"].map(time_interval_60_min)

time_dim["label_hh"] = time_dim["time_key"].map(label_hh)
time_dim["label_hh24"] = time_dim["time_key"].map(label_hh24)

time_dim["label_15min_24"] = time_dim["time_key"].map(label_15_min_24)
time_dim["label_30min_24"] = time_dim["time_key"].map(label_30_min_24)
time_dim["label_60min_24"] = time_dim["time_key"].map(label_60_min_24)

time_dim["label_15min_12"] = time_dim["time_key"].map(label_15_min_12)
time_dim["label_30min_12"] = time_dim["time_key"].map(label_30_min_12)
time_dim["label_60min_12"] = time_dim["time_key"].map(label_60_min_12)

time_dim["minute_after_hour"] = time_dim["time_key"].map(minute_after_hour)

date_dim.head()

time_dim.head(n=100)

## Export each to a CSV first so that we can use the COPY command. It's substantially more efficient.
# date_dim.to_csv(path_or_buf="./date_dim.csv", index=False)
time_dim.to_csv(path_or_buf="./time_dim.csv", index=False)

engine = create_engine('postgresql://' + username + ':' + password + '@' + host + '/' + db_name)

empty_date_dim = date_dim.copy()
empty_date_dim = empty_date_dim.drop(empty_date_dim.index[0:date_dim.shape[0]])
empty_date_dim.to_sql("date_dimension", engine, if_exists="replace", index=False)

empty_time_dim = time_dim.copy()
empty_time_dim = empty_time_dim.drop(empty_time_dim.index[0:time_dim.shape[0]])
empty_time_dim.to_sql("time_dimension", engine, if_exists="replace", index=False)

connection_string = "dbname='" + db_name + "' "
connection_string += "user='" + username + "' "
connection_string += "host='" + host + "' "
connection_string += "password='" + password + "' "

# Set up a copy statement. The %s will be replaced later
sql_statement = """
    COPY %s FROM STDIN WITH
    CSV
    HEADER
    DELIMITER AS ','
"""

def load_file(conn, table_name, primary_key, file_object):
    cursor = conn.cursor()
    cursor.copy_expert(sql=sql_statement % table_name, file=file_object)
    conn.commit()
    
    # Add add primary key, index and then vacuum 
    cursor.execute("ALTER TABLE " + table_name + " ADD  PRIMARY KEY (" + primary_key + ")")
    cursor.execute("VACUUM VERBOSE ANALYZE " + table_name)
    cursor.execute("CREATE INDEX ON " + table_name + " (" + primary_key + " ASC NULLS LAST);")
    cursor.close()

date_dim_file = open("./date_dim.csv")
date_dim_file

conn = psycopg2.connect(connection_string)
conn.autocommit = True
try:
    load_file(conn, table_name='date_dimension', primary_key="date_key", file_object=date_dim_file)
finally:
    conn.close()

time_dim_file = open("./time_dim.csv")
time_dim_file

conn = psycopg2.connect(connection_string)
conn.autocommit = True
try:
    load_file(conn, table_name='time_dimension', primary_key="time_key", file_object=time_dim_file)
finally:
    conn.close()

pokemon_info_df = pd.read_csv(filepath_or_buffer="./dimension_table_csvs/pokemon_info.csv")
empty_pokemon_pk_info = pokemon_info_df.copy()
empty_pokemon_pk_info = empty_pokemon_pk_info.drop(empty_pokemon_pk_info.index[0:empty_pokemon_pk_info.shape[0]])
empty_pokemon_pk_info.to_sql("pokemon_info", engine, if_exists="replace", index=False)

pokemon_info_dim_file = open("./dimension_table_csvs/pokemon_info.csv")
pokemon_info_dim_file

conn = psycopg2.connect(connection_string)
conn.autocommit = True
try:
    load_file(conn, table_name='pokemon_info', primary_key="pokemon_id", file_object=pokemon_info_dim_file)
finally:
    conn.close()



