# Extracting death row inmates
from lxml import html as htmlparser
from os.path import join
from os import makedirs
from urllib.parse import urljoin
import csv
import requests

DEATHROW_URL = 'http://wgetsnaps.github.io/tdcj-state-tx-us--death_row/death_row/dr_offenders_on_dr.html'
# set the file format in terms of headers
DEATHROW_TABLE_HEADERS = ['tdcj_number', 'inmate_info_url', 'last_name', 'first_name', 
                          'birthdate', 'gender', 'race', 
                         'date_received', 'county', 'date_offense']
FILE_HEADERS = DEATHROW_TABLE_HEADERS + ['date_executed', 'last_words_url']

# set up the directory/filename
DATA_DIR = join('data', 'tx-death-penalty', 'extracted')
DEST_FILENAME = join(DATA_DIR, 'texas-death-row.csv')
makedirs(DATA_DIR, exist_ok=True)

# Download and parse the table
deathrow_html = requests.get(DEATHROW_URL).text
deathrow_doc = htmlparser.fromstring(deathrow_html)
# xpath is the awesome
# http://stackoverflow.com/questions/10881179/xpath-find-all-elements-with-specific-child-node
deathrows = deathrow_doc.xpath('//table[@class="os"]/tbody/tr[td]')

# open and prepare the file for writing
wf = open(DEST_FILENAME, 'w')
csvfile = csv.DictWriter(wf, fieldnames = FILE_HEADERS)
csvfile.writeheader()

# iterate through each html row
for row in deathrows:
    # get column HTML element for each table columnf
    cols = row.xpath('td')
    # create dictionary     
    d = dict(zip(DEATHROW_TABLE_HEADERS, [td.text_content().strip() for td in cols]))
    # have to manually extract column href from second column (e.g. "Offender Information")
    href = cols[1].xpath('//a/@href')[0]
    d['inmate_info_url'] = urljoin(DEATHROW_URL, href)
    # they haven't been executed yet
    d['date_executed'] = None
    d['last_words_url'] = None
    # write to CSV
    csvfile.writerow(d)    

print("Wrote", len(deathrows), "rows to:", DEST_FILENAME)    
wf.close()



