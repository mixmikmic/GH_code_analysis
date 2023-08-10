import requests
from bs4 import BeautifulSoup

wiki_response = requests.get(url='https://en.wikipedia.org/wiki/List_of_accidents_and_incidents_involving_commercial_aircraft')

soup = BeautifulSoup(wiki_response.text, 'html.parser')

# Finding all anchor tags with a link
links = soup.find_all('a', href=True)

bolds = soup.find_all('b')
bolds[1].find('a')

example_tag = links[3]

example_tag['href']

example_tag.get('href')

print(bolds[0])
print(bolds[3])

type(wiki_response)

type(links)



