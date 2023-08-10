import requests
from bs4 import BeautifulSoup
import csv



url = 'http://www.nrc.gov/reactors/operating/list-power-reactor-units.html'

web_page = requests.get(url)
soup = BeautifulSoup(web_page.content, 'html.parser')

reactor_table = soup.find('table')

csv_file = open('reactors.csv', 'wb')
output = csv.writer(csv_file)

output.writerow(['NAME', 'LINK', 'DOCKET', 'LICENSE_NUM', 'TYPE', 'LOCATION', 'OWNER', 'REGION'])

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
    
csv_file.close()

