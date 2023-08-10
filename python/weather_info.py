import forecastio
import datetime as dt
import os

api_key = os.getenv('FORECASTIO_API_KEY') # INSERT YOUR OWN API KEY HERE
lat = 42.28
lng = -83.74
t = dt.datetime(2017,1,31,11,0,0)
response = forecastio.load_forecast(api_key, lat, lng, time=t)
forecast = response.json

forecast['hourly']['data'][0]

for hr, data in enumerate(forecast['hourly']['data']):
    print('{0:0>2}:00 Temp: {1}F, WindSpeed: {2}, Icon Summary:{3}'.format(hr,data['windSpeed'],data['temperature'],data['icon']))



