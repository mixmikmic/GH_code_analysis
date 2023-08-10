import requests
# import everything from BeautifulSoup
from BeautifulSoup import *

url = "https://hrantdavtyan.github.io/"

response = requests.get(url)
my_page = response.text
print(response)
type(my_page)

soup = BeautifulSoup(my_page)

type(soup)

print(soup)

a_tags = soup.findAll('a')

type(a_tags)

len(a_tags)

print(a_tags)

a_tag = soup.find('a')
type(a_tag)

print(a_tag)

print(a_tag.get('href'))

for i in a_tags:
    print(i.get("href"))

p_tags = soup.findAll('p')
print(p_tags)

for i in p_tags:
    print(i.text)

