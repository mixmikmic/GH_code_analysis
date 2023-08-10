from __future__ import division

from urllib2 import urlopen
from urllib2 import HTTPError
from urllib2 import URLError

from bs4 import BeautifulSoup, UnicodeDammit
from IPython.core.display import HTML
from bson import json_util


from pymongo import MongoClient
import pymongo

import json
import pprint

import numpy as np
import pandas as pd

import requests
import itertools
import datetime
import random
import os
import re


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

client = MongoClient()

# dir(client)

clicksdb = client.clicks.log
cursor = clicksdb.find()
cursor

def getjsonfrom_cursor(database):
    cursor = database.find().limit(10)
    return json_util.dumps(cursor)

clicks_json = getjsonfrom_cursor(clicksdb) 

parsing = json.loads(clicks_json)

parsingprint = pprint.PrettyPrinter()

parsingprint.pprint(parsing[1])

clicksdb.find({'cy': 'Seattle'}).count()

len(clicksdb.distinct('_id'))

browser00 = re.compile(ur'mozilla', re.IGNORECASE)
browser01 = re.compile(ur'opera', re.IGNORECASE)

print clicksdb.find({'a': browser00}).count()

print clicksdb.find({'a': browser01}).count()



























html_todo = urlopen('https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining')

# Print the whole html
# html_todo.read()

sopita = BeautifulSoup(html_todo.read(), 'lxml')

# Get header
print(sopita.h1)

# Function for retrieving errors
def getTitle(url):
    try:
        html = urlopen('https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining')
    except AttributeError as e:
        return None
    try:
        sopita = BeautifulSoup(html.read())
        title = sopita.body.h1
    except AttributeError as e:
        return none
    return title

title = getTitle('https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining')
if title == None:
    print('Title could not be found')
else:
    print(title)
    

r = requests.get('https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining')

HTML(r.content)

soup = BeautifulSoup(r.content, "lxml")

# From soup --> content is a list, adding [0] to remove brackets
# 'div', {'class': 'reflist'} --> tagName, TagAttributes 
for a in soup.findAll('div', {'class': 'reflist'})[0].findAll('a'):
    print(a['href'])


# 'a', {'class': 'external text'} --> tagName, TagAttributes 
for a in soup.find_all('a', {'class':'external text'}):
    print a['href']

hazme = requests.get('https://www.kaggle.com/account/login')
HTML(hazme.content)

access_ = os.getenv('Kaggle_u')
#access = '{}'.format(access_)

password_ = os.getenv('Kaggle_p')
#password = '{}'.format(password_)

data_url = 'https://www.kaggle.com'

kinfo = {'UserName':'{}'.format(access_), 'Password': '{}'.format(password_)}

kr_ = requests.get(data_url)

kr = requests.post(kr_.url, data=kinfo)


HTML(kr.content)













