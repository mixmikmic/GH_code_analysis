# Import required libraries
import requests
import json
from __future__ import division
import math
import csv
import matplotlib.pyplot as plt

# set key
key="be8992a420bfd16cf65e8757f77a5403:8:44644296"

# set base url
base_url="http://api.nytimes.com/svc/search/v2/articlesearch"

# set response format
response_format=".json"

# set search parameters
search_params = {"q":"Duke Ellington",
                 "api-key":key}       

# make request
r = requests.get(base_url+response_format, params=search_params)

print(r.url)

# set date parameters here

# Uncomment to test
# r = requests.get(base_url+response_format, params=search_params)
# print(r.url)

# set page parameters here

# Uncomment to test
# r = requests.get(base_url+response_format, params=search_params)
# print(r.url)

# Inspect the content of the response, parsing the result as text
response_text= r.text
print(response_text[:1000])

# Convert JSON response to a dictionary
data = json.loads(response_text)
# data

print(data.keys())

# this is boring
data['status']

# so is this
data['copyright']

# this is what we want!
# data['response']

data['response'].keys()

data['response']['meta']['hits']

# data['response']['docs']
type(data['response']['docs'])

docs = data['response']['docs']

docs[0]

import time
from random import randint

# set key
key="ef9055ba947dd842effe0ecf5e338af9:15:72340235"

# set base url
base_url="http://api.nytimes.com/svc/search/v2/articlesearch"

# set response format
response_format=".json"

# set search parameters
search_params = {"q":"Duke Ellington",
                 "api-key":key,
                 "begin_date":"20150101", # date must be in YYYYMMDD format
                 "end_date":"20151231"}

# make request
r = requests.get(base_url+response_format, params=search_params)
    
# convert to a dictionary
data=json.loads(r.text)
    
# get number of hits
hits = data['response']['meta']['hits']
print("number of hits: ", str(hits))
    
# get number of pages
pages = int(math.ceil(hits/10))
    
# make an empty list where we'll hold all of our docs for every page
all_docs = [] 
    
# now we're ready to loop through the pages
for i in range(pages):
    print("collecting page", str(i))
        
    # set the page parameter
    search_params['page'] = i
        
    # make request
    r = requests.get(base_url+response_format, params=search_params)
    
    # get text and convert to a dictionary
    data=json.loads(r.text)
        
    # get just the docs
    docs = data['response']['docs']
        
    # add those docs to the big list
    all_docs = all_docs + docs
    
    time.sleep(randint(3,5))  # pause between calls

len(all_docs)

# DEFINE YOUR FUNCTION HERE

# uncomment to test
# get_api_data("Duke Ellington", 2014)

all_docs[0]

def format_articles(unformatted_docs):
    '''
    This function takes in a list of documents returned by the NYT api 
    and parses the documents into a list of dictionaries, 
    with 'id', 'header', and 'date' keys
    '''
    formatted = []
    for i in unformatted_docs:
        dic = {}
        dic['id'] = i['_id']
        dic['headline'] = i['headline']['main']
        dic['date'] = i['pub_date'][0:10] # cutting time of day.
        formatted.append(dic)
    return(formatted) 

all_formatted = format_articles(all_docs)

all_formatted[:5]

def format_articles(unformatted_docs):
    '''
    This function takes in a list of documents returned by the NYT api 
    and parses the documents into a list of dictionaries, 
    with 'id', 'header', 'date', 'lead paragrph' and 'word count' keys
    '''
    formatted = []
    for i in unformatted_docs:
        dic = {}
        dic['id'] = i['_id']
        dic['headline'] = i['headline']['main']
        dic['date'] = i['pub_date'][0:10] # cutting time of day.
        
        # YOUR CODE HERE
        
        formatted.append(dic)
    return(formatted) 

# uncomment to test
# all_formatted = format_articles(all_docs)
# all_formatted[:5]

keys = all_formatted[1]
# writing the rest
with open('all-formated.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(all_formatted)

# YOUR CODE HERE

