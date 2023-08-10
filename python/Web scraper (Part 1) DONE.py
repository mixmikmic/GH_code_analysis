import requests
from bs4 import BeautifulSoup
import csv

url = "http://www.nrc.gov/reactors/operating/list-power-reactor-units.html"
web_page = requests.get(url)

print(web_page.text)

soup = BeautifulSoup(web_page.content, 'html.parser')
reactor_table = soup.find('table')

print(reactor_table)

data_file = open('reactors.csv', 'wb')
output = csv.writer(data_file)

output.writerow(["NAME", "LINK", "DOCKET", "LICENSE_NUM", "TYPE", "LOCATION", "OWNER", "REGION"])

test_row = reactor_table.find_all('tr')[1]
cell_list = test_row.find_all('td')

print(cell_list)

cell_list[0].contents

print(cell_list[0].contents[0].text)
print(cell_list[0].contents[0].get('href'))
print(cell_list[0].contents[2].strip())

for row in reactor_table.find_all('tr')[1:]:
    
    cell = row.find_all('td')
    
    name = cell[0].contents[0].text
    link = cell[0].contents[0].get('href')
    docket = cell[0].contents[2].strip()
    lic_num = cell[1].text
    reactype = cell[2].text
    
    location = cell[3].text.encode('utf-8')
    owner = cell[4].text.strip().encode('utf-8')
    region = cell[5].text

    output.writerow([name, link, docket, lic_num, reactype, location, owner, region])

data_file.close()

