# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, inspect, func

# Create an engine to a SQLite database file called `surfsup.sqlite`
engine = create_engine("sqlite:///hawaii.sqlite")

from sqlalchemy.orm import Session
# Connect to database to be able to run test queries
session = Session(bind=engine)

inspector = inspect(engine)
inspector.get_table_names()

# Get a list of column names and types
columns = inspector.get_columns('measurement')
for c in columns:
    print(c['name'], c["type"])

engine.execute('SELECT * FROM measurement LIMIT 5').fetchall()

engine.execute("SELECT * FROM measurement WHERE date = '2016-09-14'").fetchall()

from sqlalchemy.ext.automap import automap_base

# Reflect Database into ORM class
Base = automap_base()
Base.prepare(engine, reflect=True)
Measurement = Base.classes.measurement
Station = Base.classes.station

# Query all tobs values
results = session.query(Measurement.tobs).all()

# Convert list of tuples into normal list
tobs_values = list(np.ravel(results))
tobs_values

# Query for last 12 months of precipitation
last_12_months_precipitation = session.query(Measurement.date, Measurement.prcp).        filter(Measurement.date >= '2016-08-24').filter(Measurement.date <= '2017-08-23').order_by(Measurement.date).all()

# Set above query results to dataframe
df_last12months_precipitation = pd.DataFrame(data=last_12_months_precipitation)
df_last12months_precipitation.head(40)

df_last12months_precipitation = df_last12months_precipitation.set_index("date")
df_last12months_precipitation.head(20)

# Define labels
plt.title("Precipitation for last 12 Months")
plt.xlabel("Month")
plt.ylabel("Precipitation in inches")

# Define months for x-ticks labels
months = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]

# Define X and Y values
y = df_last12months_precipitation["prcp"].tolist()
x = np.arange(0, len(df_last12months_precipitation.index.tolist()), 1)

# Define X-tick labels (months) and their positioning
month_total = len(y)
month_step_xticks = int((month_total / 12)*1.03)
plt.ylim = max(y) + 1
tick_locations = [x+55 for x in range(1, month_total, month_step_xticks)]

# Define plot
plt.bar(x, y, width=30, color="blue", alpha=0.5, align="edge")
plt.xticks(tick_locations, months)

plt.show()

from sqlalchemy import func

# Total number of stations
totalnumber_of_stations = session.query(func.count(Station.station)).first()

# Print results of above count query
print(f"Total number of stations: {str(totalnumber_of_stations[0])}")

# Run query to verify the measurement counts by station
engine.execute("SELECT count(station), station FROM measurement GROUP BY station ORDER BY count(station) DESC").fetchall()

# Query to find the most active stations
active_stations_descending = session.query(Measurement.station, func.count(Measurement.station)).        group_by(Measurement.station).order_by(func.count(Measurement.station).desc()).all()

# Set above query results to dataframe
df_active_stations_descending = pd.DataFrame(data=active_stations_descending, columns=['Station', 'Count'])
df_active_stations_descending.head()

# Set station with highest number of observations to a variable
station_with_most_observations = df_active_stations_descending["Station"][0]
most_observations = df_active_stations_descending["Count"][0]
print(f"Station with most observations ({most_observations}): {station_with_most_observations}")

# Query for temperature counts (a) for the last year and (b) at the most active station
temperature_frequencies = session.query(Measurement.tobs).    filter(Measurement.date >= '2016-08-24').    filter(Measurement.station == station_with_most_observations).    order_by(Measurement.tobs).all()
    
temperature_frequencies

# Define the histogram from the above dataset, with 12 bins
hist, bins = np.histogram(temperature_frequencies, bins=12)

# Set bar width to the number of values between each bin
width = bins[1] - bins[0]

# Plot the bar graph from the histogram data
plt.bar(center, hist, width=width)
plt.show()

# Function "calc_temps": accepts start date and end date in the format '%Y-%m-%d' 
#  and returns the minimum, average, and maximum temperatures for that range of dates
def calc_temps(start_date, end_date):
    """TMIN, TAVG, and TMAX for a list of dates.
    
    Args:
        start_date (string): A date string in the format %Y-%m-%d
        end_date (string): A date string in the format %Y-%m-%d
        
    Returns:
        TMIN, TAVE, and TMAX
    """
    
    return session.query(func.min(Measurement.tobs), func.avg(Measurement.tobs), func.max(Measurement.tobs)).        filter(Measurement.date >= start_date).filter(Measurement.date <= end_date).all()
    
# Function "last_year_dates": accepts start date and end date in the format '%Y-%m-%d' 
#  and returns the equivalent dates for the previous year
def last_year_dates(start_date, end_date):
    """ Corresponding dates from previous year
    Args:
        start_date (string): A date string in the format %Y-%m-%d
        end_date (string): A date string in the format %Y-%m-%d
        
    Returns:
        start_date (string)
        end_date (string)
    """
    lst_start_date = start_date.split('-')
    lst_end_date = end_date.split('-')
    lastyear_start_year = int(lst_start_date[0]) - 1
    lastyear_end_year = int(lst_end_date[0]) - 1
    ly_start_date = f"{lastyear_start_year}-{lst_start_date[1]}-{lst_start_date[2]}"
    ly_end_date = f"{lastyear_end_year}-{lst_end_date[1]}-{lst_end_date[2]}"
    
    return (ly_start_date, ly_end_date)

#
# *** Define trip dates ***
#
trip_start = '2015-04-20'
trip_end = '2015-04-28'

# Call function to return average temperatures for this date range
average_trip_temps = calc_temps(trip_start, trip_end)

# Call function to grab lates from last year
(lastyear_start_date, lastyear_end_date) = last_year_dates(trip_start, trip_end)

import seaborn 

# Define standard error (max minus min temps)
yerr_val = average_trip_temps[0][2] - average_trip_temps[0][0]

# Y value is the average temperature for the trip's date range; X is zero since we only need one bar
y = [average_trip_temps[0][1]]
x = 0

# Define plot
fig, ax = plt.subplots()

# Add  labels, title and axes ticks
ax.set_ylabel("Temperature (F)", fontsize=14)
ax.set_title("Trip Average Temps", fontsize=18)

# Set the limits of the x and y axes, no tick params
ax.bar(x, y, width=.1, color="blue", yerr=yerr_val)
ax.set_xlim(-.1, .1)
ax.set_ylim(0, 100)
ax.set_xbound(lower=-.1, upper=.1)
ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') 
plt.show()

# Query to calculate sum of all 'prcp' for each weather station for previous year dates
rainfall_by_station_lastyear = session.query(Measurement.station, func.sum(Measurement.prcp)).    filter(Measurement.date >= lastyear_start_date).    filter(Measurement.date <= lastyear_end_date).    group_by(Measurement.station).    order_by(func.sum(Measurement.prcp).desc()).all()
rainfall_by_station_lastyear

query_to_run = f"SELECT station, sum(prcp) FROM measurement WHERE date >= '{lastyear_start_date}' AND date <= '{lastyear_end_date}' "            "GROUP BY station "            "ORDER BY sum(prcp) DESC"
print(query_to_run)
engine.execute(query_to_run).fetchall()



