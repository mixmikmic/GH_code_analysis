# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.
get_ipython().magic('matplotlib inline')
# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import requests
import json

#These are all the nodes we're interested in so far.
nodeList = [
    5021673,
    32417525,
    32417527,
    32417545,
    32417547,
    32417599,
    32417601,
    32417629,
    32417631,
    32417633,
    32417635
]
#This is the base URL for the PJM REST API
url = 'https://dataminer.pjm.com/dataminer/rest/public/api/markets/realtime/lmp/daily'

def splitDateTime(utchour):
    #split datetime into date and time components
    datetime_parts = utchour.split('T', 1)
    parts = dict(date = datetime_parts[0], time = datetime_parts[1].rstrip('Z'))
    return parts

from datetime import datetime, timedelta, date, time

#Using code adapted from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(start, end):
    for n in range(int((end - start).days)):
        yield start + timedelta(n)
        
def formatDate(aDate):
    return aDate.strftime('%Y-%m-%d')

def adjustTime(parts):
    dtstring = parts['date'] + ' ' + parts['time']
    dtformat = '%Y-%m-%d %H:%M:%S'
    adjusted = datetime.strptime(dtstring, dtformat) - timedelta(hours = 4)
    return adjusted

def getHour(adjustedDatetime):
    t = adjustedDatetime.time()
    tstring = t.strftime('%H:%M:%S')
    tparts = tstring.split(':', 2)
    return tparts[0]

#HERE IS THE BIG KAHUNA
#This will take a long-ass time to run (25 mins), because we have to loop over every day in the years 2008 - 2012
from datetime import date
#set up our json POST data
params_list = [
    dict(startDate = formatDate(date(2008, 1, 1)), endDate = formatDate(date(2008, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2009, 1, 1)), endDate = formatDate(date(2009, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2010, 1, 1)), endDate = formatDate(date(2010, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2011, 1, 1)), endDate = formatDate(date(2011, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2012, 1, 1)), endDate = formatDate(date(2012, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2013, 1, 1)), endDate = formatDate(date(2013, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2014, 1, 1)), endDate = formatDate(date(2014, 12, 31)), pnodeList = nodeList),
    dict(startDate = formatDate(date(2015, 1, 1)), endDate = formatDate(date(2015, 11, 1)), pnodeList = nodeList)
]

results_dict = {}

for i in range(0, len(params_list)):
    
    #make the API call
    r = requests.post(url, json = params_list[i])
    if r.status_code == requests.codes.ok:
        results_dict[i] = r.json()
    else:
        r.raise_for_status()
        
    #be nice to the API, wait 2 seconds
    time.sleep(2)

recordsList = []
for result in results_dict.values():
    
    #make a new row for each individual price
    for record in result:
        #we are only interested in Total LMP per Sam's email
        if record['priceType'] == 'TotalLMP':
            data = {}
            data['pnodeId'] = record['pnodeId']
            published = splitDateTime(record['publishDate'])
            data['publishDate'] = published['date']
            for p in record['prices']:
                utcparts = splitDateTime(p['utchour'])
                hour = getHour(adjustTime(utcparts))
                if hour == '00':
                    hour = '24'
                key = 'price_' + hour
                data[key] = p['price']
            recordsList.append(data)

#let's see what we have. It's probably obscenely huge.
#print rawdf.shape
#results_dict[0][0:3]
print len(recordsList)
#recordsList[0:2]

rawdf = pd.DataFrame(recordsList)

rawdf.to_csv('rawdf_pjm_realtime.csv')

rawdf.head()

grouped = rawdf.groupby('publishDate')
for k, v in grouped[:3]:
    print k



