import requests
from bs4 import BeautifulSoup 
import re
import pandas as pd

base_url = 'http://www.datatau.com'

#Let us use request to get the url
dataTau = requests.get(base_url)

# Check if the page has been scraped - we should see Response 200
dataTau

dataTau = open('../../data/dataTau.html', 'rb').read()

# Let us see the text content of the page
dataTau

# Start the beautifulsoup library and create a soup!
soup = BeautifulSoup(dataTau,'html.parser')

# See the pretty form HTML - Not so pretty though!
print (soup.prettify())

title_class = soup.select('td .title')

len(title_class)

title_class[0:2]

title_class[-1]

title_class = soup.select('td .title a')

len(title_class)

title_class[0]

title_class[0].get_text()

title_class[-1]

title_class = soup.select('td .title > a:nth-of-type(1)')

title_class[0].get_text()

date_class = soup.select('.subtext')

len(date_class)

date_class[0]

date_class[0].get_text()

# Let us create an empty dataframe to store the data
df = pd.DataFrame(columns=['title','date'])
df.count()

def get_data_from_tau(url):
    print(url)
    dataTau = requests.get(url)
    soup = BeautifulSoup(dataTau.content,'html.parser')
    title_class = soup.select('td .title > a:nth-of-type(1)')
    date_class = soup.select('.subtext')
    print(len(title_class),len(date_class))
    for i in range(len(title_class)-1):
        df.loc[df.shape[0]] = [title_class[i].get_text(),date_class[i].get_text()]
    print('updated df with data')
    return title_class[len(title_class) - 1]

url = base_url
for i in range(0,6):
    more_url = get_data_from_tau(url)
    url = base_url+more_url['href']

df.shape

df.head()

df.to_csv('data_tau.csv', encoding = "utf8", index = False)



