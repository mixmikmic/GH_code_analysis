# The code was removed by DSX for sharing.

import requests
import json
def print_api_resp(url,credentials):
    response = requests.get(credentials_1["url"]+":443/"+url)  
    parsed = json.loads(response.text)
    print json.dumps(parsed, indent=4, sort_keys=True)

print_api_resp("api/weather/v1/geocode/45.42/75.69/forecast/hourly/48hour.json?units=m&language=en-US",credentials_1)

print_api_resp("api/weather/v1/location/97229%3A4%3AUS/forecast/hourly/48hour.json?units=m&language=en-US",credentials_1)

print_api_resp("api/weather/v1/geocode/45.42/75.69/forecast/daily/10day.json?units=m&language=en-US",credentials_1)

print_api_resp("api/weather/v1/geocode/45.42/75.69/observations.json?units=m&language=en-US",credentials_1)

print_api_resp("api/weather/v1/geocode/34.12/-117.30/alerts.json?language=en-US",credentials_1)

print_api_resp("api/weather/v1/geocode/33.40/-83.42/almanac/daily.json?units=e&start=0112&end=0115",credentials_1)

print_api_resp("api/weather/v3/location/search?query=Atlanta&locationType=city&countryCode=US&adminDistrictCode=GA&language=en-US",credentials_1)

print_api_resp("api/weather/v3/location/point?geocode=34.53%2C-84.50&language=en-US",credentials_1)

print_api_resp("api/weather/v3/location/point?postalKey=30339%3AUS&language=en-US",credentials_1)

