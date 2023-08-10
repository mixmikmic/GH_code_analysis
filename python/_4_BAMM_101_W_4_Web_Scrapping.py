import requests
from bs4 import BeautifulSoup

url = "https://www.simplyrecipes.com/recipes/type/baking/"
response = requests.get(url,stream = False)
if response.status_code == 200:
    print("Success")
else:
    print("Failure",response.status_code)

requests.head('https://www.simplyrecipes.com/recipes/type/baking/')

searching = input("Please enter the things you want to see in a recipe: ")
url = "https://www.simplyrecipes.com/?s=" + searching

response = requests.get(url)
if response.status_code == 200:
    print("Success")
else:
    print("Failure",response.status_code)

results_page = BeautifulSoup(response.content,'lxml')
print(results_page.prettify())

all_a_tags = results_page.find_all('a')
print(type(all_a_tags))
print(all_a_tags)

div_tag = results_page.find('div')
print(div_tag.prettify())

div_tag.find('a')

div_tag.find('a').find('span')

results_page.find_all('h2',class_='entry-title')

results_page.find('h2',{'class':'entry-title'})

results_page.find('h2',{'class':'entry-title'}).get_text()

entry_title_tag = results_page.find('ul',{'class':'entry-list'}).find('a').get('href')
entry_title_tag



