# Extracting death row executions
from bs4 import BeautifulSoup
from os.path import join
from os import makedirs
from urllib.parse import urljoin
import csv
import requests
import re

EXECUTED_URL = 'http://wgetsnaps.github.io/tdcj-state-tx-us--death_row/death_row/dr_executed_offenders.html'
EXECUTED_TABLE_HEADERS = ['inmate_info_url', 'last_words_url', 'last_name', 'first_name', 
                          'tdcj_number', 'executed_age', 'executed_date', 'race',  'county']

INMATE_FIELDS_TO_EXTRACT = {
    'birthdate': 'Date of Birth', 
    'date_offense': 'Date of Offense',
    'date_received': 'Date Received',
    'gender': 'gender'    
}    


FILE_HEADERS = EXECUTED_TABLE_HEADERS + list(INMATE_FIELDS_TO_EXTRACT.keys())


# set up the directory/filename
DATA_DIR = join('data', 'tx-death-penalty', 'extracted')
DEST_FILENAME = join(DATA_DIR, 'texas-executed.csv')
makedirs(DATA_DIR, exist_ok=True)

executed_html = requests.get(EXECUTED_URL).text
executed_doc = BeautifulSoup(executed_html, 'lxml')
executed_rows = executed_doc.select('table.os tr')[1:] # skip first row of headers

wf = open(DEST_FILENAME, 'w')
csvfile = csv.DictWriter(wf, fieldnames = FILE_HEADERS, restval="")
csvfile.writeheader()

for row in executed_rows: # skip first row of table headers
    cols = row.find_all('td')[1:] # skip first column
    # create dictionary 
    d = dict(zip(EXECUTED_TABLE_HEADERS, [td.text.strip() for td in cols]))
    d['inmate_info_url'] = urljoin(EXECUTED_URL, cols[0].find('a')['href'])
    d['last_words_url'] = urljoin(EXECUTED_URL, cols[1].find('a')['href'])
    # write to CSV
    csvfile.writerow(d)

print("Wrote", len(executed_rows), 'rows in:', DEST_FILENAME)



