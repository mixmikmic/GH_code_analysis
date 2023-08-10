import json

input = '''[
  { "id" : "01",
    "status" : "Instructor",
    "name" : "Hrant"
  } ,
  { "id" : "02",
    "status" : "Student",
    "name" : "Jimmy"
  }
]'''

# parse/load string
data = json.loads(input)
# data is a usual list

type(data)

print(data)

from pprint import pprint

pprint(data)

print 'User count:', len(data), "\n"

data[0]['name']

for element in data:
    print 'Name: ', element['name']
    print 'Id: ', element['id']
    print 'Status: ', element['status'], "\n"

import pandas as pd

address = "C:\Data_scraping\JSON\sample_data.json"

my_json_data = pd.read_json(address)

my_json_data.head()

import json

with open(address,"r") as file:
    local_json = json.load(file)

print(local_json)

type(local_json)

pprint(local_json)

with open('our_json_w.json', 'w') as output:
    json.dump(local_json, output)

with open('our_json_w.json', 'w') as output:
    json.dump(local_json, output, sort_keys = True, indent = 4)

import csv, json
address = "C:\Data_scraping\JSON\sample_data.json"
with open(address,"r") as file:
    local_json = json.load(file)

with open("from_json.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ID","Name","Status"])
    for item in local_json:
        writer.writerow([item['id'],item['name'],item['status']])

