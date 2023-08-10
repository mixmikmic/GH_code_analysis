import pandas as pd
import numpy as np
import collections
import re
from pandas.tools.plotting import table
import matplotlib.pyplot as plt
from itertools import islice
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')

from dateutil.parser import parse
def datemaker(date):
    dt = parse(date)
    return dt

# Morning till 11:59 | Afternoon 12-3pm | Evening 3pm-8:59pm | Night after 9
def timeOfDay(date):
    #dt = parse(date)
    if date.hour < 12:
        return "Morning"
    if date.hour>= 12 and date.hour <= 15:
        return "Afternoon"
    if date.hour> 15 and date.hour < 21:
        return "Evening"
    else:
        return "Night"

def usergroup(userID):
    if userDict[userID].TOD.value_counts(ascending=False).index.tolist()[0] is 'Morning':
        return
    if userDict[userID].TOD.value_counts(ascending=False).index.tolist()[0] is 'Afternoon':
        return
    if userDict[userID].TOD.value_counts(ascending=False).index.tolist()[0] is 'Evening':
        return
    if userDict[userID].TOD.value_counts(ascending=False).index.tolist()[0] is 'Night':
        return

def vsearch(x):
    flag = 0
    for v in vulgar:
        if v in x:
            return True
            flag =1
    if flag == 0:
        return False

#Load the dataset
data = pd.read_csv("/Users/Rohit/Desktop/Data/test.txt", sep="\t")
mydict = {}
data.shape

data.head()

timedf = data[["AnonID", "Query", "ItemRank", "QueryTime"]]
timedf.head()
timedf['QueryDate'] = pd.to_datetime(timedf['QueryTime']).apply(lambda x: x.date())
timedf['QueryTimeStamp'] = pd.to_datetime(timedf['QueryTime']).apply(lambda x: x.time())
timedf['TOD'] = timedf['QueryTimeStamp'].apply(lambda x: timeOfDay(x))

timedf['Vulgar'] = timedf.Query.apply(vsearch)

timedf.head()

vdf = timedf[timedf['Vulgar'] == True]
vdf.ItemRank.count()/vdf.Query.count()
#Number of vulgar queries that lead to click

userlist = data.AnonID.unique()
len(userlist) #No of unique users

total = 0
for i in data.AnonID:
    if i in mydict: 
        mydict[i] += 1
    else: 
        mydict[i] = 1
    total+=1

len(mydict) #Should be the same as userlist

#Verifying dictionary is correct
sum =0
users =""
for k,v in mydict.items():
    sum+=v   
print(sum) #Should be same as shape[0]

#Changing to % of total
for k,v in mydict.items():
    mydict[k] = (v/sum) * 100

#Find the most common users in this list to analyze their moves
commonUsers = sorted(mydict.items(), key=lambda x:-x[1])[:10]
# commonuserList = [i[0] for i in commonUsers]
# commonuserFreq = [i[1] for i in commonUsers]
# plt.bar(range(len(commonuserList)), commonuserFreq, align="center")
# plt.xticks(range(len(commonuserList)) , commonuserList, rotation = 70)
# plt.show()

from difflib import SequenceMatcher

def similar(a, b):
    return round(SequenceMatcher(None, a, b).ratio(),2)

mostActive = data[data.AnonID == commonUsers[0][0]]
# mAunique = mostActive.Query.unique()
(mostActive.AnonID.count() - mostActive.ItemRank.count())/mostActive.AnonID.count()
values1 = [mostActive.ItemRank.count(),(mostActive.AnonID.count() - mostActive.ItemRank.count())]
labels1 = ['Number of Queries Without Clicked URLs', 'Number of Queries With Clicked URLs'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('QueriesWithClicks.png')
plt.show()

mAqueries = {}
for i in mostActive.Query:
    if i in mAqueries: 
        mAqueries[i] += 1
    else: 
        mAqueries[i] = 1

mAqueries = sorted(mAqueries.items(), key=lambda x:-x[1])

keyys = 0
totalsingle = 0
for i in mAqueries:
    if i[1] == 1:
        totalsingle += mostActive.loc[mostActive.Query == i[0]].Query.count()
        keyys += mostActive.loc[mostActive.Query == i[0]].ClickURL.isnull().count()

onetimeSuccess = (totalsingle - keyys)/totalsingle
onetimeSuccess

import enchant
d = enchant.Dict("en_US")
cspell=0
csearch=[]
wspell=0
wsearch=[]
for i in mAqueries:
    ssum=0
    for word in i[0].split():
        ssum+= d.check(word)
    if ssum==len(i[0].split()):
        csearch+=i[0],
        cspell+=1
    else:
        wsearch+=i[0]
        wspell+=1

values1 = [cspell,wspell]
labels1 = ['Correctly Spelled', 'Wrongly Spelled'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('SpellCheckQueries.png')
plt.show()

vulgar = ['porn', 'dick', 'sexy', 'sex', 'pussy', 'ass', 'fuck', 'sperm']
vcount = 0
allvcount = 0
for i in csearch:
    for v in vulgar:
        if v in i: 
            vcount+=1

for i in mostActive.Query:
    for v in vulgar:
        if v in i: 
            allvcount+=1

print(vcount)
print(allvcount)

values1 = [vcount/allvcount,allvcount/allvcount]
labels1 = ['Correctly spelled Vulgar Searches', 'All vulgar search attempts'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('VulgarQueries.png')
plt.show()

vcount/cspell #Out of the correctly spelled searches
values1 = [vcount/cspell,cspell/cspell]
labels1 = ['Vulgar Queries', 'All Correctly Spelled Queries'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('CorrectVulgarQueries.png')
plt.show()

(allvcount)/mostActive.Query.count() #Out of the total searches
values1 = [(allvcount)/mostActive.Query.count(),mostActive.Query.count()/mostActive.Query.count()]
labels1 = ['Vulgar Searches from Most Active User', 'All Queries from Most Active User'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('VulgarUsedByMostActiveUsers.png')
plt.show()

