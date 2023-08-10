# Import Libraries
import numpy as np
import pandas as pd
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from urllib import quote_plus

df = pd.read_csv('../data/ratings.csv',names=['dist','votes','rating','title'],encoding='mbcs')

df['year'] = df['title'].str.extract('\((\d\d\d\d)\/*\w*\) *\(*.*\)*$')
df.dropna(inplace=True)
df['year'] = df['year'].astype(np.int32)
df['shorttitle'] = df['title'].str.extract('(.*) \(\d\d\d\d\/*\w*\) *\(*.*\)*$')
df['shorttitle'] = df['shorttitle'].apply(lambda x: x.strip())

dfsub = df[df['votes']>5000]
print(len(dfsub))
dfsub.head()

title = '*batteries not included'
year = 1987
r = requests.get('http://www.omdbapi.com/?t=' + quote_plus(title) + '&y=' +str(year) + '&plot=short&r=json&tomatoes=true')

data = r.json()
print(data)

def getJSON(dfrow):
    import requests
    from urllib import quote_plus
    title = dfrow['shorttitle']
    year = dfrow['year']
    r = requests.get('http://www.omdbapi.com/?t=' + quote_plus(title) + '&y=' +str(year) + '&plot=short&r=json&tomatoes=true')

    data = r.json()
    newDBrow = pd.io.json.json_normalize(data)
    #print(newDBrow)
    return newDBrow
getJSON(dfsub.iloc[0]).columns.values

nrows = 0
newdb = pd.DataFrame()
for row,datarow in dfsub.iterrows():
    import time
    newrow=getJSON(datarow)
    if 'Error' not in newrow:
        newrow['index'] = row
        newrow.set_index('index',inplace=True)
        newdb = newdb.append(newrow)
        #print(newrow)
    nrows += 1
    time.sleep(0.1)
    if (nrows % 100 == 0):
        print('Acquired {}\n'.format(nrows))
newdb.to_csv('../data/openmdb.csv')

dfsub2 = df[df['votes']>1500]
dfsub2 = dfsub2[dfsub2['votes']<=5000]
print(len(dfsub2))
dfsub2.head()

nrows = 0
newdb2 = pd.DataFrame()
for row,datarow in dfsub2.iterrows():
    import time
    newrow=getJSON(datarow)
    if 'Error' not in newrow:
        newrow['index'] = row
        newrow.set_index('index',inplace=True)
        newdb2 = newdb2.append(newrow)
        #print(newrow)
    nrows += 1
    time.sleep(0.1)
    if (nrows % 100 == 0):
        print('Acquired {}\n'.format(nrows))
newdb2.to_csv('../data/openmdb_more.csv')

#Combine the data and look only at the films with more than 1500 votes on IMDb
newdb=pd.read_csv('../data/openmdb.csv',encoding='mbcs',index_col=0)
newdb2=pd.read_csv('../data/openmdb_more.csv',encoding='mbcs',index_col=0)
newdb3 =pd.concat([newdb,newdb2])
dfsubmore = df[df['votes']>1500]
print(len(newdb3))

print(len(dfsubmore))
dbbig = newdb3.join(dfsubmore)
print(len(dbbig))

def change_text(text):
    return text.encode('utf-8')  # assuming the encoding is UTF-8

#Get the first director and the list of directors
dbbig = dbbig[dbbig['Director'].notnull()]

dbbig['Director1']=dbbig['Director'].apply(lambda x: (str(change_text(x)).split(',')[0]))

#Create the genre sub listings including 'None' if there are no additional entries
dbsub = dbbig['Genre'].apply(lambda x: pd.Series([i for i in reversed(str(x).split(','))]))
dbsub.columns = ['Genre1', 'Genre2','Genre3']
def getGenre(x):
    if pd.notnull(x) and x != 'nan':
        return str(x).strip()
    else:
        return 'None'
dbsub['Genre1'] = dbsub['Genre1'].apply(getGenre)
dbsub['Genre2'] = dbsub['Genre2'].apply(getGenre)
dbsub['Genre3'] = dbsub['Genre3'].apply(getGenre)
dbbig = dbbig.join(dbsub)

#Create an integer rating that we can predict based on the average value
dbbig['stars'] = dbbig['rating'].apply(lambda x: int(np.round(x)))

dblearn =dbbig[['title','year','Director1','Genre1','Genre2','Genre3','stars']]
dblearn =dblearn.reindex(np.random.permutation(dblearn.index))
dblearn.to_csv('../data/movie_ratings_simple.tsv',sep='\t',index=False,na_rep='None',encoding='utf-8')

dblearn.to_csv('../data/movie_ratings_simple.csv',sep=',',index=False,na_rep='None',encoding='utf-8')

