import seaborn as  sns
import matplotlib as plt
import datetime
import pandas as pd


get_ipython().magic('matplotlib inline')

data = pd.read_csv('#solidariTEA_1.0.csv',sep='\t')

data.head()

from got3 import manager

tweetCriteria = manager.TweetCriteria().setQuerySearch('#solidariTEA').setSince("2017-12-06").setUntil("2017-12-08").setMaxTweets(800)
tweet = manager.TweetManager.getTweets(tweetCriteria)

def dataframe(tweet):
    data = pd.DataFrame(columns=['Username', 'Text', 'mentions','retweets','hastags','geo','favorites','date','month','day','hour' ])
    Columns = ['Username', 'Text', 'mentions','retweets','hastags','geo','favorites','date','month','day','hour' ]
    for t in tweet: 
        username = t.username
        text = t.text
        mentions = t.mentions
        retweets = t.retweets
        hastags = t.hashtags
        date = t.date
        geo = t.geo
        favorites = t.favorites
        month = t.date.month
        day =  t.date.day
        hour  = t.date.hour
       
        Row = [username,text,mentions,retweets,hastags,geo,favorites,date,month,day,hour]
        data = data.append(pd.DataFrame([Row], columns=Columns))
    return(data)    

data = dataframe(tweet)

data.head()

hour = data['hour'].tolist()
myplot = sns.distplot(hour, bins=23, kde=False, rug=True,axlabel='Hours(00.00 to 23.59)')

sns.jointplot(x='favorites', y='retweets', data=data)

Total_Retweets = data['retweets'].sum()
Total_favorites = data['favorites'].sum()
print ('Total retweets:', Total_Retweets)
print ('Total favorites:', Total_favorites)

data.max()

Max_Tweet = data.max()
print(Max_Tweet['Text'])
print(Max_Tweet['retweets'])
print(Max_Tweet['favorites'])

data.Text.str.split(expand=True).stack().value_counts()

data.mentions.str.split(expand=True).stack().value_counts()

data.hastags.str.split(expand=True).stack().value_counts()

for t in tweet:
    print("Username: %s" % t.username)
    print("Text: %s" % t.text)
    print("Mentions: %s" % t.mentions)
    print("Retweets: %s" % t.retweets)
    pr#
    int("Hashtags: %s\n" % t.hashtags)
    print("Date: %s\n" % t.date)
    print("Geo: %s\n" % t.geo)
    print("Favorites: %s\n" % t.favorites)
    print('          ')

