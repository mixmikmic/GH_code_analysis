import requests

page = requests.get("http://dataquestio.github.io/web-scraping-pages/simple.html")
print page

# status code that starts with 2 == good; starts with 4 or 5 == bad
print page.status_code

# view html content
page.content

from bs4 import BeautifulSoup
soup = BeautifulSoup(page.content, 'html.parser')

soup

print soup.prettify()

# collect all elements at top level of age
print list(soup.children)

print [type(item) for item in soup.children]

html = list(soup.children)[2]
print list(html.children)
print len(list(html.children))
print "\n"
body = list(html.children)[3]
print body

print list(body.children)

p = list(body.children)[1]
p

# In one line
soup.find_all('p')

soup.find_all('p')[0].get_text()

# finding only the first method
soup.find('p')

page = requests.get("http://dataquestio.github.io/web-scraping-pages/ids_and_classes.html")
soup = BeautifulSoup(page.content, 'html.parser')

print soup.prettify()

# paragraphs with class 'outer-text'
soup.find_all('p', class_='outer-text')
soup.select("p.outer-text")

# any tag with class outer-text
soup.find_all(class_='outer-text')

soup.find_all(id="second")

page = requests.get("http://forecast.weather.gov/MapClick.php?lat=37.7772&lon=-122.4168#.WKHxhxIrKRs")
soup = BeautifulSoup(page.content, 'html.parser')

print soup.prettify()

seven_day = soup.find(id='seven-day-forecast-list')
print seven_day.prettify()

forecast_stuff = soup.find_all(class_='tombstone-container')
print forecast_stuff
print len(forecast_stuff)

tonight = forecast_stuff[0]
print tonight.prettify()

period = tonight.find(class_='period-name').get_text()
shortdsc = tonight.find(class_='short-desc').get_text()
temp = tonight.find(class_ = 'temp temp-high').get_text()
print period
print shortdsc
print temp

img = tonight.find('img')
desc = img['title']
print desc

period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
periods

sd_tags = seven_day.select(".tombstone-container .short-desc")
sd_tags = [sd.get_text() for sd in sd_tags]
sd_tags

temp = seven_day.select(".tombstone-container .temp")
temp = [t.get_text() for t in temp]
temp

descs = seven_day.select(".tombstone-container img")
descs = [d['title'] for d in descs]
descs
#descs = [d["title"] for d in seven_day.select(".tombstone-container img")]

import pandas as pd
weather = pd.DataFrame({
        "period": periods, 
        "short_desc": sd_tags, 
        "temp": temp, 
        "desc":descs
    })

temp_nums = weather["temp"].str.extract("(?P<temp_num>\d+)")
weather["temp_num"] = temp_nums.astype('int')
temp_nums

weather

weather['temp_num'].mean()



