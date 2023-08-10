from BeautifulSoup import *
import requests

url = "https://careercenter.am/ccidxann.php"

response = requests.get(url)
page = response.text
soup = BeautifulSoup(page)

tables = soup.findAll("table")

len(tables)

print(tables[0])

my_table = tables[0]
rows = my_table.findAll('tr')

data_list = []
for i in rows:
    columns = i.findAll('td')
    for j in columns:
        data_list.append(j.text)

print(data_list)

type(data_list)

data_list[:10]

