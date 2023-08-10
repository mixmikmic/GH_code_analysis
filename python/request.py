# just for presentation in notebooks
from pprint import pprint as print

import requests
import csv


csv_source = 'http://data.gov.uk/data/resource/nhschoices/Hospital.csv'

csv_delimiter = '\t'

response = requests.get(csv_source)

raw = response.text.splitlines()

reader = csv.DictReader(raw, delimiter=csv_delimiter)

data = []

for row in reader:
    data.append(row)

print(data[:2])

import requests


json_source = 'https://api.fixer.io/latest'

response = requests.get(json_source)

data = response.json()

print(data)

import requests
from pyquery import PyQuery


html_source = 'https://www.gov.uk/bank-holidays'

response = requests.get(html_source)

document = PyQuery(response.text)

answer = document('.calendar:first tbody tr:first td:first').text()

print('{} is the next bank holiday in the UK.'.format(answer))



