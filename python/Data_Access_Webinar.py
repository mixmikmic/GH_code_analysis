import csv

reader =csv.reader(open('image03.txt',"r"), delimiter = '\t')

dm = []
for row in reader:
    for element in row:
        if element.startswith('s3'):
            #print (element)
            dm.append(element)

#Download Manager as a Webservice

import requests
import xml.etree.ElementTree as ET
from getpass import getpass

url = 'https://ndar.nih.gov/DataManager/dataManager'
username = input('Enter your NIMH Data Archives username:')
password = getpass('Enter your NIMH Data Archives password:')
package = input('Enter package ID:') #ex: 108728, 108335, 108772

payload = ('<?xml version="1.0" ?>\n'+
           '<S:Envelope xmlns:S="http://schemas.xmlsoap.org/soap/envelope/">\n' +
           '<S:Body> <ns3:QueryPackageFileElement\n' +
               'xmlns:ns4="http://dataManagerService"\n' +
               'xmlns:ns3="http://gov/nih/ndar/ws/datamanager/server/bean/jaxb"\n'+
               'xmlns:ns2="http://dataManager/transfer/model">\n' +
                   '<packageId>' + package +'</packageId>\n' +
                   '<associated>true</associated>\n' +
               '</ns3:QueryPackageFileElement>\n' +
           '</S:Body>\n' +
           '</S:Envelope>')


headers = {
    'Content-Type': "text/xml"
    }

r = requests.request("POST", url, data=payload, headers=headers)

root = ET.fromstring(r.text)

path = root.findall(".//path")

ws = []
for element in path:
    file = 's3:/'+ element.text
    print(file)
    ws.append(file)

import cx_Oracle
hostname = 'mindarvpc.cqahbwk3l1mb.us-east-1.rds.amazonaws.com'
port = 1521
sid = 'ORCL'

import getpass
username = input('Enter your miNDAR username:')#'username_package_id'
password = getpass.getpass('Enter your miNDAR password:')
dsnStr = cx_Oracle.makedsn(hostname, port, sid )
db = cx_Oracle.connect(user=username, password=password, dsn=dsnStr)

#here we can retrieve the s3 URLs for image_file and data_file assoicated with each subject from the image03 table. 
#there is also an S3_LINKS table that lists all the s3 URLs available. 

import pandas as pd
import pandas.io.sql as psql

query = """SELECT subjectkey, image_file, data_file2
            FROM IMAGE03
            WHERE data_file2 IS NOT NULL"""
            
c = db.cursor()
c.execute(query)
data_files = pd.DataFrame(c.fetchall())
data_files.columns = [rec[0] for rec in c.description]

print(data_files)

miNDAR = []
for file in data_files['IMAGE_FILE']:
    miNDAR.append(file)

from getpass import getpass

url = 'https://ndar.nih.gov/DataManager/dataManager'
username = input('Enter your NIMH Data Archives username:')
password = getpass('Enter your NIMH Data Archives password:')

from nda_aws_token_generator import *
generator = NDATokenGenerator(url)
token = generator.generate_token(username, password)

print('aws_access_key_id=%s\n'
      'aws_secret_access_key=%s\n'
      'security_token=%s\n'
      'expiration=%s\n' 
      %(token.access_key,
        token.secret_key,
        token.session,
        token.expiration)
      )

# Pull image out of S3

file = input("Enter S3 URL:")
#s3://NDAR_Central_2/submission_11013/002000001590/scanVisit__0020__0002/MRI__0001/B0_phase1/Native/Original__0001/DICOM.tar.gz

import boto.s3.connection
from urllib.parse import urlparse

cf = boto.s3.connection.OrdinaryCallingFormat()
conn = boto.connect_s3( token.access_key,
                        token.secret_key,
                        security_token=token.session,
                        calling_format=cf)

bucket = urlparse(file).netloc
key = urlparse(file).path
bucket_object = conn.get_bucket(bucket)
s3_object = boto.s3.key.Key(bucket_object)
s3_object.key = key
name = key.split('/')
name = name[-1]
byte_data = s3_object.get_contents_to_filename(name) #edit this file name to actually match s3

#lists = [dm, ws, miNDAR]
#will save only first five files in the list
count = 1
for file in dm:
    bucket = urlparse(file).netloc
    key = urlparse(file).path
    bucket_object = conn.get_bucket(bucket)
    s3_object = boto.s3.key.Key(bucket_object)
    s3_object.key = key
    name = key.split('/')
    name = name[-1]
    byte_data = s3_object.get_contents_to_filename(name) #edit this file name to actually match s3
    print(name)
    count += 1
    if count > 5:
        break

import requests
import json
from getpass import getpass
#NDAR_INVRT663MBL
#NDAR_INVEP756TR3
#NDARBV344JLX

username = input("What is your NDA username:")
password = getpass("What is your NDA password:")
guid = input("What GUID would you like to access data from:")
#r = requests.get("https://ndar.nih.gov/api/guid/{}/data?short_name=image03".format(guid), 
#                 auth=requests.auth.HTTPBasicAuth(username, password),
#                 headers={'Accept': 'application/json'})

r = requests.get("https://stage.nimhda.org/api/guid/{}/data?short_name=image03".format(guid), 
                 auth=requests.auth.HTTPBasicAuth(username, password),
                 headers={'Accept': 'application/json'})

guid_data = json.loads(r.text)
print(guid_data)

# Extract experiment IDs from response

experiments = []
ages = []
for age in guid_data['age']:
    age_value = age['value']
    for row in age['dataStructureRow']:
        for element in row['dataElement']:
            if element['name']=='EXPERIMENT_ID':
                if element['value'] not in experiments:
                    experiments.append(element['value'])

for experiment in experiments:
    print('experiment: {}'.format(experiment))

# In previous 2 slides, have identified some fMRI, EEG, or Eye Tracking data; show how to retreive experimental details.

query = input("Enter your experiment ID:")
r = requests.get("https://ndar.nih.gov/api/experiment/{}".format(query),
                 headers={'Accept':'application/json'})

experiment = json.loads(r.text)
print(experiment)

# Pull out image files from response
image_files = []
ages = []
for age in guid_data['age']:
    age_value = age['value']
    for row in age['dataStructureRow']:
        for link in row['links']['link']:
            if link['rel']=='data_file':
                image_files.append(link['href'])
                ages.append(age_value)
guid_list = []
for i,image in enumerate(image_files):
    print("age:{}, url:{}".format(ages[i],image))
    guid_list.append(image)
    

# Data Element Full Search

import requests
import json

query_phrase = input("Enter a description phrase:") #example: ABCD, depression, autism

r = requests.post('https://ndar.nih.gov/api/search/nda_sw_removal/dataelement/full?size=10', 
                  headers={'Accept':'application/json'}, data= query_phrase)

search_results = json.loads(r.text)

data_elements = search_results['datadict']
print ('Total data structures:', data_elements['total'])
print()

for i in data_elements['results']: #filter the data structures
    if i['_score'] > 0.33:
        if query_phrase.lower() in i['description'].lower():
            print ('Name:', i['name'])
            print ('id:', i['id'])
            print ('Notes:', i['notes'])
            print ('Description:', i['description'])
            print ('Score:', i['_score'])
            print ('Data Structures:')
            for k in i['dataStructures']:
                print(k)
            print ('Total Data Structures:', i['total_data_structures'])
            print()

# Data Element Search

description = input("Enter a description to query:")
query = {'description': description}
r = requests.post("https://stage.nimhda.org/api/search/nda_sw_removal/dataElementSearch?size=20", 
                  data=json.dumps(query),
                  headers={'content-type':'application/json'})
element_results = json.loads(r.text)
for result in element_results['dataElements']: 
    print("score:{}\nname:{}\ndescription:{}\n".format(result['score'], result['name'], result['description']))

# Here is a programmatic example searching by collection
import requests
import json


class collectionLink():

    def __init__(self, title, id):
            self.id = id
            self.title = title
            self._repr_html_()

    def _repr_html_(self):
        collection_link = 'https://ndar.nih.gov/edit_collection.html?id={}'.format(self.id)
        html = ['<a href="{}">{}</a>'.format(collection_link, self.title)]
        return ''.join(html)

    
query = input("Enter your query phrase:")
r = requests.post("https://ndar.nih.gov/api/search/nda_sw_removal/collection/full", query)
collections = json.loads(r.text)
print("\n")
for result in collections['collection']['results']:
    display(collectionLink(result['title'],result['id'])) 

# Progromatically retrieve data from the dictionary service
# Example data structure shortname: image03

import requests
import json
shortname = input('Enter a Data Structure shortname:')
r = requests.get('https://ndar.nih.gov/api/datadictionary/v2/datastructure/{}'
                 .format(shortname),
                  headers={'Accept':'application/json'})
structure = json.loads(r.text)

# Get Data Structure change history

r = requests.get('https://ndar.nih.gov/api/datadictionary/v2/datastructure/{}/changes'
                 .format(shortname),
                  headers={'Accept':'application/json'})

changes = json.loads(r.text)

# Show data structure elements that are required, or potentially required (conditional)

for element in structure['dataElements']:
    if element['required'] in ['Required','Conditional']:
        print('elementInfo: {}\n'.format(element))

# Show element name for all data elements with type as 'file'

for element in structure['dataElements']:
    if element['type'] == 'File':
        print('elementName: {}'.format(element['name']))

# Get data element info and changes
from IPython.display import display

class changeHistoryTable():
    
    def __init__(self, list):
        self.list = list
        self.headers = ['id','changeDescription','changedDate','elementName','newValue','oldValue','shortName']
        self._repr_html_()
    
    def _repr_html_(self):

        html = ["<table width=100%>"]
        html.append("<thead><tr>")
        for header in self.headers:
            html.append("<td>{}</td>".format(header))
        html.append("</tr></thead><tbody>")      

        for row in self.list:
            html.append("<tr>")
            for header in self.headers:
                html.append("<td>{}</td>".format(row[header]))
            html.append("</tr>")
        html.append("</tbody></table>")
        return ''.join(html)

change_list = []
        
for element in structure['dataElements']: #search data elements in a data structure
    if element['type'] == 'File':
        r = requests.get('https://ndar.nih.gov/api/datadictionary/v2/dataelement/{}' 
                 .format(element['name']),
                  headers={'Accept':'application/json'})
        elementInfo = json.loads(r.text) #information about the data element
        
        r = requests.get('https://ndar.nih.gov/api/datadictionary/v2/dataelement/{}/changes'
                .format(element['name']),
                headers={'Accept':'application/json'})
        changes = json.loads(r.text) #information about the data element changes
        try:
            change_list.extend(changes['list'])
        except KeyError:
            print('No changes for elementName {}'.format(element['name']))

display(changeHistoryTable(change_list))

# Data Element Data Structure Search
# Returns a list of data structures for data element alphabetically
# Example data elements: gender, subjectkey

dataelement = input("Enter a data element name:")

r = requests.get('https://ndar.nih.gov/api/search/nda_sw_removal/dataElementDataStructures?name={}'.format(dataelement), 
                  headers={'Accept':'application/json'})

element = json.loads(r.text)

for data in element['dataElements']:
    for structure in data['dataStructures']:
        print("id:{}\ncategory:{}\nshortName:{}\nsubjectCount:{}\n".format(structure['id'], structure['category'], structure['shortName'], structure['subjectCount']))

