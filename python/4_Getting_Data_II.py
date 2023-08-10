import numpy as np
import pandas as pd

# Need requests to connect to the URL, json to convert JSON to dict
import requests, json
import pprint

# joining words in the address by a "+"
add = "UpGrad, Nishuvi building, Anne Besant Road, Worli, Mumbai"
split_address = add.split(" ")
address = "+".join(split_address)
print(address)


api_key = "AIzaSyBXrK8md7uaOcpRpaluEGZAtdXS4pcI5xo"

url = "https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}".format(address, api_key)
r = requests.get(url)

# The r.text attribute contains the text in the response object
print(type(r.text))
print(r.text)

# converting the json object to a dict using json.loads()
r_dict = json.loads(r.text)

# the pretty printing library pprint makes it easy to read large dictionaries
pprint.pprint(r_dict)

# The dict has two main keys - status and results
r_dict.keys()

pprint.pprint(r_dict['results'])

lat = r_dict['results'][0]['geometry']['location']['lat']
lng = r_dict['results'][0]['geometry']['location']['lng']

print((lat, lng))

# Input to the fn: Address in standard human-readable form
# Output: Tuple (lat, lng)

api_key = "AIzaSyBXrK8md7uaOcpRpaluEGZAtdXS4pcI5xo"

def address_to_latlong(address):
    # convert address to the form x+y+z
    split_address = address.split(" ")
    address = "+".join(split_address)
    
    # pass the address to the URL
    url = "https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}".format(address, api_key)
    
    # connect to the URL, get response and convert to dict
    r = requests.get(url)
    r_dict = json.loads(r.text)
    lat = r_dict['results'][0]['geometry']['location']['lat']
    lng = r_dict['results'][0]['geometry']['location']['lng']
    
    return (lat, lng)
    

# getting some coordinates
print(address_to_latlong("UpGrad, Nishuvi Building, Worli, Mumbai"))
print(address_to_latlong("IIIT Bangalore, Electronic City, Bangalore"))

# Importing addresses file
add = pd.read_csv("addresses.txt", sep="\t", header = None)
add.head()

# renaming the column
add = add.rename(columns={0:'address'})
add.head()

add.head()['address'].apply(address_to_latlong)

import PyPDF2

# reading the pdf file
pdf_object = open('../Data/animal_farm.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_object)

# Number of pages in the PDF file
print(pdf_reader.numPages)

# get a certain page's text
page_object = pdf_reader.getPage(5)

# Extract text from the page_object
print(page_object.extractText())



