import requests
from bs4 import BeautifulSoup
import unicodecsv as csv
import time

# text-finding function goes here
def finder(list_to_search, term):
    for item in list_to_search:
        if term.upper() in item.upper():
            return item.split(':')[1].strip()

url = "http://www.nrc.gov/reactors/operating/list-power-reactor-units.html"
main_page = requests.get(url)
soup = BeautifulSoup(main_page.content, 'html.parser')

reactors_table = soup.find('table')

scraped = []

for row in reactors_table.find_all('tr')[1:]:
    cells = row.find_all('td')
    reactor_name = cells[0].contents[0].text
    link = 'http://www.nrc.gov' + cells[0].contents[0].get('href')
    docket = cells[0].contents[2]
    license = cells[1].text
    reactor_type = cells[2].text
    location = cells[3].text
    owner = cells[4].text
    region = cells[5].text
    # add steps to the loop here
    # get the individual reactor page with requests
    reactor_page = requests.get(link)
    # run the response through BeautifulSoup so that it can be navigated
    reactor_soup = BeautifulSoup(reactor_page.content, 'html.parser')
    # isolate the table cell with the text we want to pick over and then split it up on line breaks
    cell = reactor_soup.find_all('td')[1]
    lines = cell.text.split('\n')
    # use the new function to grab the megawattage, vendor and containment type
    mwt = finder(lines, 'mwt')
    vendor = finder(lines, 'vendor')
    containment_type = finder(lines, 'containment')
    # print an informational status message to yourself
    print('Just scraped data for {}...'.format(reactor_name))
    # add these to the list that will ultimately be written to CSV
    scraped.append([reactor_name, link, docket, license, reactor_type, location, owner, region, mwt, vendor, containment_type])
    # IMPORTANT: pause for a couple of seconds between page requests
    time.sleep(2)

with open('reactor_data_extended.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    # add the new columns to the header row
    writer.writerow(['reactor_name', 'link', 'docket', 'license', 'reactor_type', 'location', 'owner', 'region', 'mwt', 'vendor', 'containment_type'])
    writer.writerows(scraped)

