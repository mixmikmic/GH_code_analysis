from bs4 import BeautifulSoup
import requests

# Sites to exclude from our trending news URL collection
exclusions = ['google.com','youtube.com','wikipedia.org','blogspot.com']
prefix = 'http://'

def include_url(url):
    for excl in exclusions:
        if url.find(excl) > 0:
            return False
    return True

# Fetch the page content and extract the links
def fetch_links(url):
    response = requests.get(prefix + url)
    page_content = response.text
    soup = BeautifulSoup(page_content, "lxml")
    links = soup.find_all('a')
    return links

import os

hdfs_location = '/Users/Projects/Current/Notebook'

linkset = set()
links = fetch_links('news.google.com')
    
# Collect the article links applying the URL filters
for link in links:
    href = link.get('href')
    if str(href).startswith(prefix) and include_url(str(href)):
        linkset.add(link.get('href').strip())
        # print str(href)
        
# Store links in HDFS
outfile = open(hdfs_location + os.path.sep + 'links' + os.path.sep + 'links.txt', "wb")
outfile.write("\n".join(linkset))

print 'Links harvested: ', str(len(linkset))

# Take 100 links for the demo
links100 = list(linkset)[:100]

# Code to store data in MongoDB using PyMongo client
from pymongo import MongoClient
from bson import json_util

def connect_to_db():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.dndemo
    return db.dailynews

import json

with open('mashape_key.txt', 'r') as keyfile:
   mashape_key = keyfile.read().rstrip()
    
# Please add your own Data Ninja API Mashape key here -->
# mashape_key = <your-mashape-key>
        
smartcontent_url = 'https://smartcontent.dataninja.net/smartcontent/tag'
headers = {'Content-Type': 'application/json', 
           'Accept': 'application/json',
           'X-Mashape-User': 'Newsbot',
           'X-Mashape-Key': mashape_key}

# If you are using AWS API Gateway, please add the X-API-Key: <your-AWS-key> 
# in place of 'X-Mashape-Key': mashape_key and use the following link to access 
# the service: https://api.dataninja.net/smartcontent/tag

def fetch_smartcontent(link):
    payload = {'url': link, 'max_size': 10}
    response = requests.post(smartcontent_url, headers=headers, data=json.dumps(payload))
    return response.json()

data = fetch_smartcontent('http://www.macrumors.com/roundup/macbook-pro/')

# Display the JSON output from Smart Content
print json.dumps(data, indent=4)

def write_to_db(data, db):
    return db.insert_one(json_util.loads(data)).inserted_id

def write_to_hdfs(data, location, filename):
    outname = location + os.path.sep + filename
    outfile = open(outname, 'w')
    outfile.write(data)
    outfile.close()
    # return hdfs.write(data, location)

# Dispay the extracted text from Smart Content
print data['text']

import json

# Call the Smart Content service and collect the article text into a list
documents = []

# Create a MongoDB connection
db = connect_to_db()

con_index = 0
cat_index = 0
for link in linkset:    
    data = fetch_smartcontent(link)
    if 'text' in data and len(data['text']) > 100:
        documents.append(data['text'])
        doc_id = write_to_db(json.dumps(data), db)
        if 'concept_list' in data:
            for concept in data['concept_list']:    
                write_to_hdfs(json.dumps(concept), hdfs_location + os.path.sep + 'concepts', 
                              'concept_' + str(con_index) + '.json')
                con_index += 1
        if 'category_list' in data:
            for category in data['category_list']:    
                write_to_hdfs(json.dumps(category), hdfs_location + os.path.sep + 'categories', 
                              'category_' + str(cat_index) + '.json')
                cat_index += 1        

print 'Documents in collection: ', str(len(documents))

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().setAppName('dataninja-pyspark')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def dndemo_spark():
    # A JSON dataset in HDFS.
    # The path can be either a single text file or a directory storing text files.
    concepts = sqlContext.read.json(hdfs_location + os.path.sep + 'concepts')
    concepts.printSchema()
    concepts.registerTempTable('concepts')
    
    categories = sqlContext.read.json(hdfs_location + os.path.sep + 'categories')
    categories.printSchema()
    categories.registerTempTable('categories')

# Run this only once!
# You can only have once SparkContext and SqlContext
dndemo_spark()    

count = sqlContext.sql('SELECT count(*) as num_concepts FROM concepts')
print count.show()

count = sqlContext.sql('SELECT count(*) as num_categories FROM categories')
print count.show()

trending_con = sqlContext.sql('SELECT concept_title, sum(score) as total_score ' +
                              'FROM concepts GROUP BY concept_title ORDER BY total_score desc')

trending_cat = sqlContext.sql('SELECT category_title, sum(score) as total_score ' +
                              'FROM categories GROUP BY category_title ORDER BY total_score desc')

# print trending.show()

from IPython.display import display, HTML
from tabulate import tabulate
import pandas as pd

display(trending_con)
display(trending_cat)

df_con = trending_con.toPandas().head(n=40)
df_cat = trending_cat.toPandas().head(n=40)

print 'Top 40 trending concepts:'
print tabulate(df_con)

print 'Top 40 trending categories'
print tabulate(df_cat)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

df_con.plot(x='concept_title', y='total_score', kind='bar', title='Trending Concepts', color='green', figsize=(20,10))
df_cat.plot(x='category_title', y='total_score', kind='bar', title='Trending Categories', color='orange', figsize=(20,10))



