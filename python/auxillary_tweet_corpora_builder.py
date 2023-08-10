import pandas as pd

df = pd.read_csv("data/tweets/disaster-tweets.csv")
print df.columns

keywords = df["keyword"].unique()
print keywords

import re

keywords = keywords[1:]

cleaned_keywords = [re.sub('%20', ' ', keyword) for keyword in keywords]

print cleaned_keywords

import tweepy

consumer_key = "IUZ7bZtjQmhtbh36FY4RtIqY4"
consumer_secret = "fEYROmCU5WTInSQOYoqLLo2x5CiagQnMNu2oLEPVoUraIfh4Cq"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

new_tweets_df = pd.DataFrame(columns = ["text"])

new_tweets = []
for keyword in cleaned_keywords:
    results = api.search(q = keyword)
    for result in results:
        new_tweets.append(result.text)

new_tweets

result.text

result.text.encode('utf-8')

len(new_tweets)

new_tweets[0]

new_tweets_df = pd.DataFrame(data = new_tweets, columns = ["text"])

import datetime

new_tweets_df.to_excel(excel_writer = 'data/' + datetime.datetime.now().strftime("%I-%M%p_%B-%d-%Y") + ".xls")

new_tweets_df.to_csv(path_or_buf = 'data/' + datetime.datetime.now().strftime("%I-%M%p_%B-%d-%Y") + ".csv")



match = re.search('^(?:(w+)\%20(w+)|(w+))$','hell0%20jim')

match = re.search('^([a-z]+)%20([a-z]+)$','hello%20jim')

match = re.search('^(?:([a-z]+)%20([a-z]+)|([a-z]+))$','hello%20jim')

print match.group(1)
print match.group(2)
print match.group(3)
if match.group(1):
    print "bob"

from twython import Twython

APP_KEY = "IUZ7bZtjQmhtbh36FY4RtIqY4"
APP_SECRET = "fEYROmCU5WTInSQOYoqLLo2x5CiagQnMNu2oLEPVoUraIfh4Cq"

twitter = Twython(APP_KEY, APP_SECRET)





print(twitter.search(q='ablaze'))

results = twitter.cursor(twitter.search, q='python')
for result in results:
    print result

import tweepy

consumer_key = "IUZ7bZtjQmhtbh36FY4RtIqY4"
consumer_secret = "fEYROmCU5WTInSQOYoqLLo2x5CiagQnMNu2oLEPVoUraIfh4Cq"
#access_token = '***'
#access_token_secret = '***'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


results = api.search(q="ablaze")

for result in results:
    print result.text

results[-1].text

import os
os.path.dirname(__file__)

results[-1]

results = api.search(q = 'ablaze')
for result in results:
    new_tweets.append(result)

results[0]

results[0].author



