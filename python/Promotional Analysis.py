import pandas as pd
import numpy as np
import collections
import re
import matplotlib.pyplot as plt
from itertools import islice
from dateutil.parser import parse
from pandas.tools.plotting import table
import datetime
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')

def URLcleaner(a):
    if 'http://' in a:
        a = a.replace('http://', '')
    if len(a.split('.'))>2:
        return a.split('.')[1].split('.')[0]
    else:
        return a.split('.')[0]

def timeOfDay(date):
    #dt = parse(date)
    if date.hour < 12:
        return "1Morning"
    if date.hour>= 12 and date.hour <= 15:
        return "2Afternoon"
    if date.hour> 15 and date.hour < 21:
        return "3Evening"
    else:
        return "4Night"

from datetime import datetime, date
def time2date(x):
    return datetime.combine(date.today(), x) 

from difflib import SequenceMatcher

def similar(a, b):
    return round(SequenceMatcher(None, a, b).ratio(),2)

import enchant
d = enchant.Dict("en_US")

def spellchk(i):
    wsum = 0 
    for word in i.split():
        if d.check(word):
            wsum = wsum +1
    if wsum == len(i.split()):
        return True
    else:
        return False

def deltaformat(x):
    x = str(x)
    if 'days' in x:
        return x.split("days")[1].split('.')[0]
    elif '.' in x:
        return x.split("days")[1].split('.')[0]
    else:
        return x

#Satisfaction Levels
def numRanks(df):
    x =df[df.ItemRank<=2.0].ItemRank.count()/df.Query.count() 
    return x

#Load the dataset
data = pd.read_csv("/Users/Rohit/Desktop/Data/test.txt", sep="\t")
data.head(10)

querylist = data.Query.unique()
len(querylist)

# fig.savefig('asdf.png')

userlist = data.AnonID.unique()
len(userlist) #No of unique users

commonusers = data.groupby(['AnonID'])['Query'].count().to_frame()
commonusers= commonusers.sort_values(by='Query', ascending=False)
commonusers.head(20).plot(kind='bar', color='purple')
plt.savefig('CommonUsers.png')

#data['QuerySpellCheck'] = data.Query.apply(lambda x: spellchk(x))

#data[data.AnonID==479].head()

queries = data.Query
faq = pd.DataFrame({'Query': queries.value_counts().index, 'Count':queries.value_counts()})
faq.index = range(faq.shape[0])
f = faq.head()
f.index = f.Query
f = f[['Count']]

f.plot(kind = 'barh', color = "peru")
plt.savefig('MostSearchedQueries.png')

toomany = data[data.ItemRank >5.0]
initial = data[data.ItemRank<=5.0]

oftenAsked = set(initial.Query) & set(toomany.Query)
stillTrying= toomany.loc[toomany['Query'].isin(oftenAsked)]

st = stillTrying.groupby(['Query'])['ItemRank'].max()
mostAttempts = st.to_frame()
mostAttempts['Queries'] = list(mostAttempts.index)

mostAttempts = mostAttempts.reset_index(drop=True)
mostAttempts= mostAttempts.sort_values(by='ItemRank', ascending=False)
mostAttempts.head(10)
# mA.plot(kind='barh')
# plt.savefig('')

rankFreq = pd.DataFrame(data.groupby(['ItemRank'])['Query'].count())
rankFreq = rankFreq.sort_values(by='Query', ascending=False).head(20)
rankFreq.plot(kind = 'barh', color='mediumvioletred', legend=False)
plt.title('Most Clicked Item Rank')
plt.savefig('HighestRankFreq')

# For Most Useful 
rankDict = {}
for rank in mostAttempts.ItemRank:
    if rank in rankDict: 
        rankDict[rank] += 1
    else: 
        rankDict[rank] = 1

rankFreq = sorted(rankDict.items(), key=lambda x:-x[1])[:10]
rankFreq = pd.DataFrame(rankFreq)
rankFreq.index = rankFreq[0]
rankFreq = rankFreq[[1]]
rankFreq.plot(kind = 'barh', color='mediumvioletred', legend=False)
plt.title('Most Clicked Item Rank')
plt.savefig('HighestRankFreqBestUser')

initial['Domain'] = initial.ClickURL.apply(URLcleaner)
frequrl = initial.groupby(['Domain'])['Domain'].count().to_frame()
frequrl['DomainName'] = list(frequrl.index)
frequrl['Count'] = frequrl['Domain']
frequrl= frequrl[['DomainName', 'Count']]
frequrl= frequrl.sort_values(by='Count', ascending=False)
frequrl = frequrl.reset_index(drop=True)
frequrl.head(10)

frequrl.index = frequrl.DomainName
frequrl.head(10).plot(kind='bar', color='coralblue')
plt.savefig('CommonURLS.png')

timedf = data[["AnonID", "QueryTime"]]
timedf.head()
timedf['QueryDate'] = pd.to_datetime(timedf['QueryTime']).apply(lambda x: x.date())
timedf['QueryTimeStamp'] = pd.to_datetime(timedf['QueryTime']).apply(lambda x: x.time())
timedf['TOD'] = timedf['QueryTimeStamp'].apply(lambda x: timeOfDay(x))

clickless = data[data.ClickURL.isnull()]
clickless['QueryTimeStamp'] = pd.to_datetime(clickless['QueryTime']).apply(lambda x: x.time())
clickless['TOD'] = clickless['QueryTimeStamp'].apply(lambda x: timeOfDay(x))

sufferingUsers = clickless.groupby(['TOD'])['AnonID'].count().to_frame()
timing = timedf.groupby(['TOD'])['AnonID'].count().to_frame()
timing['Suffering_Users'] = sufferingUsers.AnonID
timing.plot(kind='bar', color=['yellowgreen','tomato'])
plt.savefig("SufferingUsers")

newdmin = timedf.groupby(['AnonID','QueryDate'])['QueryTimeStamp'].min()

newdmax = timedf.groupby(['AnonID','QueryDate'])['QueryTimeStamp'].max()

ndmin = pd.DataFrame(newdmin)
ndmax = pd.DataFrame(newdmax)
ndmin['Min'] = ndmin['QueryTimeStamp']
ndmax['Max'] = ndmax['QueryTimeStamp']
ndmin = ndmin[['Min']]
ndmax = ndmax[['Max']]

newd = pd.concat([ndmax, ndmin], axis=1, join='inner')
newd['TimeSpent'] = newd['Max'].apply(time2date) - newd['Min'].apply(time2date)

pd.DataFrame(newd.TimeSpent.describe())

userDict= {}
for user in userlist:
    userDict[user]= data[data.AnonID==user]

userUsage={}
singleUser=0
for user in userDict.keys():
    if numRanks(userDict[user])>0.8:
        if userDict[user].shape[0]==1:
            singleUser= singleUser+1
        userUsage[user]=numRanks(userDict[user])

plt.figure()
values1 = [(data.shape[0]/data.shape[0]),(len(userUsage)/data.shape[0])] 
labels1 = ['All Users', 'Targetable Audience(Promotion Opportunities)'] 
colors1 = ['lightskyblue', 'lightcoral'] 
plt.axis('equal')
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.savefig('TargetAudience.png')
plt.show()
values2 = [(len(userUsage)/len(userUsage)), (singleUser/len(userUsage))] 
labels2 = ['Targetable Audience', 'First time Users (Retention Promotion)'] 
# colors2= ['yellowgreen', 'mediumpurple']
plt.axis('equal')
explode1 = (0, 0.1)
plt.pie(values2, labels=labels2, autopct='%.2f', explode=explode1, colors=colors1)
plt.savefig('RententionPlans.png')
plt.show()

cleandata = data.dropna()
plt.figure()
values1 = [data.shape[0]/data.shape[0],cleandata.shape[0]/data.shape[0]] 
labels1 = ['NaN Queries', 'Queries With Results'] 
colors1 = ['mediumpurple','lightskyblue'] 
plt.pie(values1, labels=labels1, autopct='%.2f', colors=colors1)
plt.axis('equal')
plt.savefig('NaNQueries.png')
plt.show()

clickless['QueryDay'] = pd.to_datetime(timedf['QueryTime']).apply(lambda x: x.date())

NaNdays = clickless.groupby(['QueryDay'])['AnonID'].count().to_frame()
alldays = timedf.groupby(['QueryDate'])['AnonID'].count().to_frame()
NaNdays['Date'] = list(NaNdays.index)
alldays['Date'] = list(alldays.index)
compDays = pd.merge(NaNdays, alldays, on='Date')
compDays['NanRequests'] = compDays['AnonID_x']
compDays['AllRequests'] = compDays['AnonID_y']
compDays = compDays[['Date', 'AllRequests','NanRequests']]
compDays = compDays.sort_values(by='NanRequests', ascending=False)

compDays.head(20).plot(x='Date', kind='bar', color=['greenyellow', 'orangered'])
plt.savefig('NanDays')

target = data2.ItemRank
predvar= data2[['AnonID', 'Hours', 'TOD_code']]
predictors=predvar.copy()
predictors['TOD_code']=preprocessing.scale(predictors['TOD_code'].astype('float64'))

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

from sklearn.linear_model import LassoLarsCV
model=LassoLarsCV(cv=3, precompute=False).fit(pred_train,tar_train)
dict(zip(predictors.columns, model.coef_))

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
clustervar = data2.copy()

clustervar['Hours']= clustervar.QueryTimeStamp.apply(lambda x: date2hours(x))

clustervar = clustervar[['ItemRank', 'Hours']]
clustervar['ItemRank']=preprocessing.scale(clustervar['ItemRank'].astype('float64'))
clustervar['Hours']=preprocessing.scale(clustervar['Hours'].astype('float64'))

clustervar.head()

clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

C = clustervar.as_matrix

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

model3=KMeans(n_clusters=4)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Item Rank')
plt.ylabel('Time of Day')
plt.show()







