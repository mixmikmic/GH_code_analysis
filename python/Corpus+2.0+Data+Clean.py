import pandas as pd
import xml.etree.cElementTree as ET #XML Parser
from lxml import etree #ElementTree and lxml allow us to parse the XML file.
import requests #make request to server
import time #pause loop
import io
import numpy as np
import re
import sys

#unzipping the file downloaded from corpus and placing into a data folder
#will be helpful later in using XML file names

import zipfile
zip_ref = zipfile.ZipFile("data/OldBaileyCorpus2.zip", 'r')
zip_ref.extractall("data/")
zip_ref.close()

#testing 1 file to check the tree structure

xml_file = 'data/OldBaileyCorpus2/OBC2/OBC2-17200427.xml'
tree = ET.ElementTree(file=xml_file)
tree

#testing the root of the file

root = tree.getroot()
print(root)

for child in root:
    print (child.tag, child.attrib)

#parsing through data to check tags and attributes

iterator = tree.getiterator()
for element in iterator:
    print(element.tag)
    print(element.attrib)
    print()

#printing all names in the XML file

for element in iterator:
    if element.tag == 'persName':
        print(element.text)

#getting text between paragraph tags, to extract the text

for element in iterator:
    if element.tag == "p":
        text = print(''.join(list(element.itertext())))
        
pd.DataFrame(text)

list_names = ([f.filename for f in zip_ref.filelist])
text = list_names[3:]

trial_numbers = []
for trial in text:
    number = re.findall(r'\d{4,}', trial)
    cleaned = str(number).strip("[]").strip("''")
    trial_numbers.append(cleaned)

trial_numbers

removed_empty = list(filter(None, trial_numbers))

# ignore, testing different methods
def xml2df(xml_data):
    root = tree.getroot() # element tree
    all_records = []
    for i, child in enumerate(root):
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
            all_records.append(record)
    return pd.DataFrame(all_records)

##MAIN ONE HAVING ISSUES WITH
from lxml import etree as et
import pandas as pd

trees = et.parse(xml_file)

d = []

for i in trees.xpath('//p'):     # ITERATE THROUGH ROOT'S CHILDREN
    inner = {}
    for elem in i.xpath('persName'):       # ITERATE THROUGH ROOT'S DESCENDANTS PER CHILD
        if len(elem.text.strip()) > 0:     # KEEP ONLY NODES WITH NON-ZERO LENGTH TEXT
            inner[elem.tag] = elem.text

    d.append(inner)



df = pd.DataFrame(d)
df

def table_of_trials(xml_file_name):
    file = ET.ElementTree(file = xml_file_name)
    iterate = file.getiterator()
    i = 1
    table = pd.DataFrame()
    for element in iterate:
        if element.tag == "p":
            try:
                t = element.attrib['type']
                if t not in labels:
                    table[t] = val
            except Exception:
                pass

            try:
                val = [element.attrib['value']]
                if t not in labels:
                    table[t] = val
                elif t+num not in labels:
                    table[t+num] = val
                elif t+num in labels:
                    num = str(i+1)
                    table[t+num] = val
            except Exception:
                pass
            labels = list(table.columns.values)
            num = str(i)
    return table
                
            

table_of_trials("data/OldBaileyCorpus2/OBC2/OBC2-17200602.xml")

def table_of_cases(xml_file_name):
    file = ET.ElementTree(file = xml_file_name)
    iterate = file.getiterator()
    i = 1
    table = pd.DataFrame()
    for element in iterate:
        if element.tag == "persName":
            t = element.attrib['type']
            try:
                val = [element.attrib['value']]
                if t not in labels:
                    table[t] = val
                elif t+num not in labels:
                    table[t+num] = val
                elif t+num in labels:
                    num = str(i+1)
                    table[t+num] = val
            except Exception:
                pass
            labels = list(table.columns.values)
            num = str(i)

    return table

table_of_cases("data/OldBaileyCorpus2/OBC2/OBC2-17200602.xml")

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

table = pd.DataFrame()
for i in removed_empty:
    raw_data = 'data/OldBaileyCorpus2/OBC2/OBC2-'+i+'.xml'
    data = table_of_trials(raw_data)
    table = table.append(data, ignore_index=True)
table

table.to_csv('oldbaily.csv')

oldbaily = pd.read_csv('oldbaily.csv')

oldbaily

oldbaily = oldbaily.drop(['Unnamed: 0', 'collection', 'collection1', 'collection2','date1', 'date2','uri', 'uri1', 'uri2', 'year1', 'year2'], axis = 1)

list(oldbaily)

oldbaily

oldbaily['age'].unique()

oldbaily['age'] = oldbaily['age'].replace('twelve Years of Age', 12).replace('nine', 9).replace('Ten next February', 9).replace('seventeen years of age', 17).replace("eleven", 11).replace("eleven Years", 11).replace("ten years old", 10).replace("thirteen or fourteen", 13.5).replace("a little above 14 years old", 14).replace("fifteen years of age", 15).replace("14 years of age", 14).replace("thirteen years of age", 13).replace("thirteen next July",12).replace("17 years od age", 17).replace("sixty-eight years", 68).replace("not 12 years old", 11).replace("nineteen", 19).replace('twenty', 20).replace("13 years", 13).replace('nine years of age',9).replace("seventeen years", 17).replace("24 years of age", 24).replace("about sixteen years of age", 16).replace("13 and 14", 13.5).replace("between nine and ten years of age", 9.5).replace("between 13 and 14", 14.5).replace("between nine and ten years of age", 9.5).replace("twenty years of age", 20).replace("between twelve and thirteen years of age", 12.5).replace("between 13 and 14", 13.5).replace("something more than twenty years of age", 20).replace('fourteen', 14).replace("sixteen", 16).replace('twelve', 12).replace("eighteen", 18).replace("fourteen years old", 14).replace("Nineteen", 19).replace("14 years old", 14).replace('seven years old', 7).replace("sixty-three", 63).replace("fifteen", 15).replace('eighteen or nineteen years of age', 18.5).replace("thirty", 30).replace("(30)", 30).replace("twelve Years of Age", 12).replace("10 years old", 10).replace("11 Years old", 11).replace("Sixteen", 16).replace("22 Years of Age", 22).replace("27 Years of Age", 27).replace('17 years of age', 17).replace("'16'", 16).replace("thirteen", 13).replace("ten", 10).replace("almost sixteen years of age", 16).replace(' (30)', 30).replace("10, 11 or 12", 11)

oldbaily['age'].unique()

##changing ages to a numeric value
oldbaily[['age']] = oldbaily[['age']].apply(pd.to_numeric)
oldbaily['age'].unique()

oldbaily['age'].mean()

oldbaily

oldbaily["name"] = oldbaily["given"].map(str) + " " + oldbaily["surname"] #combing given name and surname
oldbaily['date'] =  pd.to_datetime(oldbaily['date'], format='%Y%m%d') #changing date column into timedate

oldbaily

oldbaily = oldbaily.drop(['age', 'age1', 'age2'], axis = 1)

oldbaily['name1'] = oldbaily['given1'] + " " + oldbaily['surname1']
oldbaily['name2'] = oldbaily['given2'] + " " + oldbaily['surname2']

oldbaily
#text, date, offence, gender
#beautiful soup



