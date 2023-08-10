import pandas as pd
from pydrill.client import PyDrill

get_ipython().magic('matplotlib inline')

#Get a connection to the Apache Drill server
drill = PyDrill(host='localhost', port=8047)

#Get Written questions data - may take some time!
stub='http://lda.data.parliament.uk'.strip('/')

#We're going to have to call the API somehow
import requests


##To make thinks more efficient if we do this again, cache requests
#!pip3 install requests_cache
#import requests_cache
#requests_cache.install_cache('parlidata_cache', backend='sqlite')


#Get data from URL
def getURL(url):
    print(url)
    r=requests.get(url)
    print(r.status_code)
    return r

#Download data - if there is more, get it
def loader(url):
    items=[]
    done=False
    r=getURL(url)
    while not done:
        items=items+r.json()['result']['items']
        if 'next' in r.json()['result']:
            r=getURL(r.json()['result']['next']+'&_pageSize=500')
        else: done=True
    return items


url='{}/{}.json?session={}'.format(stub,'commonswrittenquestions','2015/16')
items=loader(url)


#Save the data
import json
with open('writtenQuestions.json', 'w') as outfile:
    json.dump(items, outfile)

#What does the whole table look like?
q=''' SELECT * from dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/writtenQuestions.json` LIMIT 3'''
drill.query(q).to_dataframe()

#Try to select a column
q='''
SELECT j.tablingMember._about AS memberURL 
FROM dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/writtenQuestions.json` j LIMIT 3
'''
drill.query(q).to_dataframe()

#Try to select an item from a list in a column
q='''
SELECT tablingMemberPrinted[0]._value AS Name 
FROM dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/writtenQuestions.json` LIMIT 3
'''
drill.query(q).to_dataframe()

#Get a dataframe of all the member URLs - so we can get the data fro each from the Parliament data API
q='''
SELECT DISTINCT j.tablingMember._about AS memberURL 
FROM dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/writtenQuestions.json` j
'''
memberIds = drill.query(q).to_dataframe()
memberIds.head()

#The URLs in the written question data donlt actually resolve - we need to tweak them
#Generate a set of members who have tabled questions that have been answered
#Note that the identifier Linked Data URL doesn't link... so patch it...
members= ['{}.json'.format(i.replace('http://','http://lda.')) for i in memberIds['memberURL']]

#Preview the links
members[:3]

#Download the data files into a data directory
get_ipython().system('mkdir -p data/members')
for member in members:
    get_ipython().system('wget -quiet -P data/members {member}')

get_ipython().system('ls data/members')

#Preview one of the files
get_ipython().system('head data/members/1474.json')

q=''' SELECT j.`result`.primaryTopic.gender._value AS gender,
j.`result`._about AS url
FROM dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/data/members` j'''
membersdf=drill.query(q).to_dataframe()
membersdf.head()

#Lets reverse the URL to the same form as in the written questions - then we can use this for a JOIN
membersdf['fixedurl']=membersdf['url'].str.replace('http://lda.','http://').str.replace('.json','')
#Save the data as a CSV file
membersdf.to_csv('data/members.csv',index=False)
get_ipython().system('head data/members.csv')

#Now find the gender of a question asker - join a query over the monolithic JSON file with the CSV file
q=''' SELECT DISTINCT j.tablingMember._about AS memberURL, m.gender
FROM dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/writtenQuestions.json` j 
JOIN dfs.`/Users/ajh59/Dropbox/parlidata/notebooks/data/members.csv` m
ON j.tablingMember._about = m.fixedurl
LIMIT 3'''
drill.query(q).to_dataframe()

#Let's see if we can modify the URL in the spearate JSON files so we can join with the monolithic file
q=''' SELECT DISTINCT j.tablingMember._about AS memberURL,
m.`result`.primaryTopic.gender._value AS gender,
m.`result`._about AS url
FROM dfs.`{path}/writtenQuestions.json` j 
JOIN dfs.`{path}/data/members` m
ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
LIMIT 3'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')
drill.query(q).to_dataframe()

q=''' SELECT COUNT(*) AS Number,
m.`result`.primaryTopic.gender._value AS gender
FROM dfs.`{path}/writtenQuestions.json` j 
JOIN dfs.`{path}/data/members` m
ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
GROUP BY m.`result`.primaryTopic.gender._value'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')
drill.query(q).to_dataframe()

q=''' SELECT COUNT(*) AS Number, j.tablingMemberPrinted[0]._value AS Name,
m.`result`.primaryTopic.gender._value AS gender
FROM dfs.`{path}/writtenQuestions.json` j 
JOIN dfs.`{path}/data/members` m
ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
GROUP BY m.`result`.primaryTopic.gender._value, j.tablingMemberPrinted[0]._value 
'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')

drill.query(q).to_dataframe().head()

q='''
SELECT AVG(Number) AS average, gender
FROM (SELECT COUNT(*) AS Number, j.tablingMemberPrinted[0]._value AS Name,
    m.`result`.primaryTopic.gender._value AS gender
    FROM dfs.`{path}/writtenQuestions.json` j 
    JOIN dfs.`{path}/data/members` m
    ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
    GROUP BY m.`result`.primaryTopic.gender._value, j.tablingMemberPrinted[0]._value )
GROUP BY gender
'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')

drill.query(q).to_dataframe()

q='''
SELECT AVG(Number) AS average, party
FROM (SELECT COUNT(*) AS Number, j.tablingMemberPrinted[0]._value AS Name,
    m.`result`.primaryTopic.party._value AS party
    FROM dfs.`{path}/writtenQuestions.json` j 
    JOIN dfs.`{path}/data/members` m
    ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
    GROUP BY m.`result`.primaryTopic.party._value, j.tablingMemberPrinted[0]._value )
GROUP BY party
'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')

dq=drill.query(q).to_dataframe()
dq['average']=dq['average'].astype(float)
dq

dq.set_index('party').sort_values(by='average').plot(kind="barh");

q='''
SELECT AVG(Number) AS average, party, gender
FROM (SELECT COUNT(*) AS Number, j.tablingMemberPrinted[0]._value AS Name,
    m.`result`.primaryTopic.party._value AS party,
    m.`result`.primaryTopic.gender._value AS gender
    FROM dfs.`{path}/writtenQuestions.json` j 
    JOIN dfs.`{path}/data/members` m
    ON j.tablingMember._about = REGEXP_REPLACE(REGEXP_REPLACE(m.`result`._about,'http://lda.','http://'),'\.json','')
    GROUP BY m.`result`.primaryTopic.party._value, m.`result`.primaryTopic.gender._value, j.tablingMemberPrinted[0]._value )
GROUP BY party, gender
'''.format(path='/Users/ajh59/Dropbox/parlidata/notebooks')

dq=drill.query(q).to_dataframe()
dq['average']=dq['average'].astype(float)
dq

dq.set_index(['party','gender']).sort_values(by='average').plot(kind="barh");

dq.sort_values(by=['gender','average']).set_index(['party','gender']).plot(kind="barh");

dqp=dq.pivot(index='party',columns='gender')
dqp.columns = dqp.columns.get_level_values(1)
dqp

dqp.plot(kind='barh');



