import requests
from bs4 import BeautifulSoup
import csv
import time

def finder(a_list, some_value):
    for item in a_list:
        if some_value.upper() in item.upper():
            return item.split(':')[1].strip()

url = 'http://www.nrc.gov/reactors/operating/list-power-reactor-units.html'

web_page = requests.get(url)
soup = BeautifulSoup(web_page.content, 'html.parser')

reactor_table = soup.find('table')

csv_file = open('reactors_more.csv', 'wb')
output = csv.writer(csv_file)

output.writerow(['NAME', 'LINK', 'DOCKET', 'LICENSE_NUM', 'TYPE', 'LOCATION', 'OWNER', 'REGION', 'MWT', 'CONTAINMENT'])

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

    # Add the new steps for this loop below
    print('Fetching details for {}...'.format(name))
    detail_page = requests.get('http://www.nrc.gov' + link)

    detail_soup = BeautifulSoup(detail_page.content, 'html.parser')
    
    new_data = detail_soup.find_all('td')[1]
    data_list = new_data.text.split('\n')

    mwt = finder(data_list, 'licensed mwt')
    containment = finder(data_list, 'containment')

    output.writerow([name, link, docket, lic_num, reactype, location, owner, region, mwt, containment])

    time.sleep(2)

csv_file.close()
print('All done!')

