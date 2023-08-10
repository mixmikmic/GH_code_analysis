import glob
import os
import json
from pathlib import Path
from collections import Counter

# prepare the path to news in year 1993
p = Path(os.getcwd())
archive_path = str(p.parent) + '/data/archive/1993/*'

# get the persons in keyword of those news and add them to person_list
files = glob.glob(archive_path)
person_list = []
for file in files:
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        python_data = json.load(f)
        docs = python_data['response']['docs']
        for doc in docs:
            if 'keywords' in doc:
                keywords = doc['keywords']
                for key in keywords:
                    if key['name'] == 'persons':
                        person_list.append(key['value'])
print(len(person_list))

# count the frequency of those names and order them
name_counter = Counter()
for name in person_list:
    name_counter.update([name])
for name, time in name_counter.most_common(10):
    print(name+': '+ str(time))

# write those persons into CSV file in popularity-descending order
import csv
output_path = 'ana_1/important_persons.csv'
with open(output_path, 'w') as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["PERSON","TIMES_OF_OCCURANCE"])
    writer.writeheader()
    for name, time in name_counter.most_common(50):
        writer.writerow({'PERSON': name,'TIMES_OF_OCCURANCE': str(time)})



