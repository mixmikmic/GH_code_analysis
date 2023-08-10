import requests
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

# Get dat
URL = ("http://forecast.weather.gov/MapClick.php?lat=25.7748&lon=-80.1977")
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast")
forecast_items = seven_day.find_all(class_="tombstone-container")
first = forecast_items[0]
print(first.prettify())

period = first.find(class_="period-name").get_text()
short_desc = first.find(class_="short-desc").get_text()
temp = first.find(class_="temp").get_text()

print(period)
print(short_desc)
print(temp)

img = first.find("img")
desc = img['title']
print(desc)

period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
periods

cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
lats = [40.7146, 34.0535, 41.8843, 29.7606, 33.4483, 39.9522, 29.4246, 32.7157, 32.7782, 37.3387]
lons = [-74.0071, -118.2453, -87.6324, -95.3697, -112.0758, -75.1622, -98.4946, -117.1617, -96.7954, -121.8854]

n_cities = len(cities)

periods = None
all_temps = []

for i in range(n_cities):
    print('Extracting data of: %s' % cities[i])
    URL = "http://forecast.weather.gov/MapClick.php?lat=" + str(lats[i]) + "&lon=" + str(lons[i])
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    seven_day = soup.find(id="seven-day-forecast")
    forecast_items = seven_day.find_all(class_="tombstone-container")
    period_tags = seven_day.select(".tombstone-container .period-name")

    #Let's save only the periods and their corresponding temperatures
    periods = [pt.get_text() for pt in period_tags]
    #short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
    temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
    #descs = [d["title"] for d in seven_day.select(".tombstone-container img")]
    
    all_temps.append(temps)

weather = pd.DataFrame(data=all_temps, index=cities, columns=periods)
print(weather.shape)
weather.head()

#Extract the numberical value of the temperature
for period in periods:
    weather[period] = weather[period].str.extract("(?P<temp_num>\d+)", expand=False)
    weather[period] = weather[period].astype('float')
weather.head()

weather = weather.T
weather.head()

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(10, 7))
fig = sns.boxplot(data=weather, ax=ax, orient='h')
fig.set_title('Temperature Range')
fig.set_xlabel('Temp (Fahrenheit)')
fig.set_ylabel('Cities')
plt.show()

