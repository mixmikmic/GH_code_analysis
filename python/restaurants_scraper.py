# scrape restaurants from chicagonow.com

import urllib
from bs4 import BeautifulSoup
url = 'http://www.chicagonow.com/chicago-food-snob/2013/12/chicagos-top-85-restaurants/'
chicago=[]
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html,"html.parser")
for item in soup.findAll('p'):
    try:
        chicago.append(item.find('a').getText())
    except:
        chicago.append(None)

# slice the restaurants name from the scraped list

chicago_dict = chicago[9:88]

# convert the name list into dataframe

import pandas as pd
df_chicago = pd.DataFrame(chicago_dict, columns={'restaurant'})

# add location and star columns

df_chicago['location'] = 1
df_chicago['star'] = 0

# store raw data

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

df_chicago.to_csv('chicago.csv', index=False)

# scrape restaurants from timeout.com

nyc=[]
for i in range(1,11):
    url = 'https://www.timeout.com/newyork/en_US/paginate?page_number={}&pageId=35907&folder=&zone_id=1202678'.format(i)
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html,"html.parser")
    for item in soup.findAll('div', {'class':'feature-item__column'}):
        try:
            nyc.append(item.find('h3').getText())
        except:
            nyc.append(None)

# clean the restaurant list (remove null values)

for x in nyc:
    if x==None:
        nyc.remove(x)

# convert restaurant list into dataframe

df_nyc = pd.DataFrame(nyc, columns={'restaurant'})

# clean restaurants name

df_nyc.restaurant = df_nyc.restaurant.apply(lambda x: x.strip())

# add location and star columns

df_nyc['location'] = 2
df_nyc['star'] = 0

# store raw data

df_nyc.to_csv('nyc.csv')

# scrape restaurants from sfchronicle.com

url = 'http://projects.sfchronicle.com/2016/top-100-restaurants/'
san=[]
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html,"html.parser")
for item in soup.findAll('div', {'class':'column restaurant-index'}):
    try:
        san.append(item.find('h5').getText())
    except:
        san.append(None)

# convert restaurant list into dataframe

df_san = pd.DataFrame(san, columns={'restaurant'})

# add location and star columns

df_san['location'] = 3
df_san['star'] = 0

# store raw data

df_san.to_csv('san.csv', index=False)

# scrape restaurants from washingtonian.com

url = 'https://www.washingtonian.com/2016/02/08/100-very-best-restaurants/'
dc=[]
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html,"html.parser")

for item in soup.findAll("tr"):
        dc.append(item.find('span',{'class':'name'}))

# convert restaurant list into dataframe

df_dc = pd.DataFrame(dc, columns=['restaurant'])

# drop null values

df_dc.dropna(inplace=True)

# clean restaurants name

df_dc.restaurant = df_dc.restaurant.apply(lambda x: str(x)[19:].replace('</span>', ''))
df_dc.restaurt = df_dc.restaurant.apply(lambda x: x.strip())

# store raw data

df_dc.to_csv('dc.csv', index=False)



