from datetime import datetime as dt, timedelta
from darksky import forecast

def convert_to_unix(time):
    return time.timestamp()

def add_week_in_unix(start):
    seconds_in_minute = 60
    
    if not isinstance(start, int):
        print("GET YOUR STARTING TIME INTO UNIX FORMAT FIRST!")
    
    return start + timedelta(days=7).total_seconds()

Johns_api_key = '99870465df482cbe2825dbd41af11496'
Auss_api_key = '80307cf49421357d3b006ce2205e8d58'

starting_date = 978346800
weeks_to_look = 893 # match row count of weekly_diseases output

berlin = (52.520008, 13.404954)

def load_data_from_location(location, weeks_to_look, starting_date, api_key):    
    temperature_per_week = []
    windspeed_per_week = []
    humidity_per_week = []
    pressure_per_week = []

    date = starting_date

    for week in range(weeks_to_look):
        try:
            location_at_week = forecast(api_key, location[0], location[1], 
                                        time=date, 
                                        units='si')
        except:
            print("Failed to access API!")
        
        try:
            temperature_per_week.append(location_at_week.temperature)
        except:
            temperature_per_week.append(None)
        try:
            windspeed_per_week.append(location_at_week.windSpeed)
        except:
            windspeed_per_week.append(None)
        try:
            humidity_per_week.append(location_at_week.humidity)
        except:
            humidity_per_week.append(None)
        try:
            pressure_per_week.append(location_at_week.pressure)
        except:
            pressure_per_week.append(None)
        
        print('LOADED:', dt.fromtimestamp(date).strftime('%Y-%m-%d'), 'call number:', week+1)
        
        date = round(add_week_in_unix(date))
            
    return temperature_per_week, windspeed_per_week, humidity_per_week, pressure_per_week

temp, wind, humid, press = load_data_from_location(berlin, weeks_to_look, starting_date, Auss_api_key)

import numpy as np

temp_a = np.asarray(temp)
wind_a = np.asarray(wind)
humid_a = np.asarray(humid)
press_a = np.asarray(press)

print(temp_a[:5])
print(wind_a[:5])
print(humid_a[:5])
print(press_a[:5])

weather_data = np.vstack((temp_a, wind_a))
weather_data = np.vstack((weather_data, humid_a))
weather_data = np.vstack((weather_data, press_a))

print(weather_data[0][:5])
print(weather_data[1][:5])
print(weather_data[2][:5])
print(weather_data[3][:5])

import pickle as pk

pk.dump(weather_data, open("weather_data.pkl", "wb"))
np.save('weather_data_np', weather_data)

