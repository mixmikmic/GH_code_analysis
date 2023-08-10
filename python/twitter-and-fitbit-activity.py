import os
import json
import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import zipfile
import pytz
import io
import sys
from textblob import TextBlob
import emoji
from ohapi import api


# sets the axis label sizes for seaborn
rc={'font.size': 14, 'axes.labelsize': 14, 'legend.fontsize': 14.0, 
    'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
sns.set(rc=rc)

# THIS CODE BELOW IS COPIED FROM TWARXIV.ORG AS IT ALREADY DOES EXACTLY WHAT WE WANT FOR READING IN THE DATA

# READ JSON FILES FROM TWITTER ARCHIVE!

def check_hashtag(single_tweet):
    '''check whether tweet has any hashtags'''
    return len(single_tweet['entities']['hashtags']) > 0


def check_media(single_tweet):
    '''check whether tweet has any media attached'''
    return len(single_tweet['entities']['media']) > 0


def check_url(single_tweet):
    '''check whether tweet has any urls attached'''
    return len(single_tweet['entities']['urls']) > 0


def check_retweet(single_tweet):
    '''
    check whether tweet is a RT. If yes:
    return name & user name of the RT'd user.
    otherwise just return nones
    '''
    if 'retweeted_status' in single_tweet.keys():
        return (single_tweet['retweeted_status']['user']['screen_name'],
                single_tweet['retweeted_status']['user']['name'])
    else:
        return (None, None)


def check_coordinates(single_tweet):
    '''
    check whether tweet has coordinates attached.
    if yes return the coordinates
    otherwise just return nones
    '''
    if 'coordinates' in single_tweet['geo'].keys():
        return (single_tweet['geo']['coordinates'][0],
                single_tweet['geo']['coordinates'][1])
    else:
        return (None, None)


def check_reply_to(single_tweet):
    '''
    check whether tweet is a reply. If yes:
    return name & user name of the user that's replied to.
    otherwise just return nones
    '''
    if 'in_reply_to_screen_name' in single_tweet.keys():
        name = None
        for user in single_tweet['entities']['user_mentions']:
            if user['screen_name'] == single_tweet['in_reply_to_screen_name']:
                name = user['name']
                break
        return (single_tweet['in_reply_to_screen_name'], name)
    else:
        return (None, None)


def create_dataframe(tweets):
    '''
    create a pandas dataframe from our tweet jsons
    '''

    # initalize empty lists
    utc_time = []
    longitude = []
    latitude = []
    hashtag = []
    media = []
    url = []
    retweet_user_name = []
    retweet_name = []
    reply_user_name = []
    reply_name = []
    text = []
    # iterate over all tweets and extract data
    for single_tweet in tweets:
        utc_time.append(datetime.strptime(single_tweet['created_at'],
                                                   '%Y-%m-%d %H:%M:%S %z'))
        coordinates = check_coordinates(single_tweet)
        latitude.append(coordinates[0])
        longitude.append(coordinates[1])
        hashtag.append(check_hashtag(single_tweet))
        media.append(check_media(single_tweet))
        url.append(check_url(single_tweet))
        retweet = check_retweet(single_tweet)
        retweet_user_name.append(retweet[0])
        retweet_name.append(retweet[1])
        reply = check_reply_to(single_tweet)
        reply_user_name.append(reply[0])
        reply_name.append(reply[1])
        text.append(single_tweet['text'])
    # convert the whole shebang into a pandas dataframe
    dataframe = pd.DataFrame(data={
                            'utc_time': utc_time,
                            'latitude': latitude,
                            'longitude': longitude,
                            'hashtag': hashtag,
                            'media': media,
                            'url': url,
                            'retweet_user_name': retweet_user_name,
                            'retweet_name': retweet_name,
                            'reply_user_name': reply_user_name,
                            'reply_name': reply_name,
                            'text': text
    })
    return dataframe


def read_files(zip_url):
    tf = tempfile.NamedTemporaryFile()
    print('downloading files')
    tf.write(requests.get(zip_url).content)
    tf.flush()
    zf = zipfile.ZipFile(tf.name)
    print('reading index')
    with zf.open('data/js/tweet_index.js', 'r') as f:
        f = io.TextIOWrapper(f)
        d = f.readlines()[1:]
        d = "[{" + "".join(d)
        json_files = json.loads(d)
    data_frames = []
    print('iterate over individual files')
    for single_file in json_files:
        print('read ' + single_file['file_name'])
        with zf.open(single_file['file_name']) as f:
            f = io.TextIOWrapper(f)
            d = f.readlines()[1:]
            d = "".join(d)
            tweets = json.loads(d)
            df_tweets = create_dataframe(tweets)
            data_frames.append(df_tweets)
    return data_frames


def create_main_dataframe(zip_url='http://ruleofthirds.de/test_archive.zip'):
    print('reading files')
    dataframes = read_files(zip_url)
    print('concatenating...')
    dataframe = pd.concat(dataframes)
    dataframe = dataframe.sort_values('utc_time', ascending=False)
    dataframe = dataframe.set_index('utc_time')
    dataframe = dataframe.replace(to_replace={
                                    'url': {False: None},
                                    'hashtag': {False: None},
                                    'media': {False: None}
                                    })
    return dataframe

user = api.exchange_oauth2_member(os.environ.get('OH_ACCESS_TOKEN'))

has_twitter = False
has_fitbit = False

# get our download URLs
for entry in user['data']:
    if entry['source'] == "direct-sharing-70":
        twitter_data_url = entry['download_url']
        has_twitter = True
    if entry['source'] == "direct-sharing-102":
        fitbit_data_url = entry['download_url']
        has_fitbit = True

if not has_twitter:
    print("YOU NEED TO HAVE SOME TWITTER DATA IN YOUR ACCOUNT TO USE THIS NOTEBOOK")
    print("GO TO http://twarxiv.org TO UPLOAD IT")

if not has_fitbit:
    print("YOU NEED TO HAVE SOME FITBIT DATA IN YOUR ACCOUNT TO USE THIS NOTEBOOK")
    print("GO TO https://www.openhumans.org/activity/fitbit-connection/ TO UPLOAD IT")
    
# read the twitter data
twitter_data = create_main_dataframe(zip_url=twitter_data_url)

# read the fitbit data
fitbit_data_raw = requests.get(fitbit_data_url).content
fitbit_data = json.loads(fitbit_data_raw)

date = []
steps = []

for year in fitbit_data['tracker-steps'].keys():
    for entry in fitbit_data['tracker-steps'][year]['activities-tracker-steps']:
        date.append(entry['dateTime'])
        steps.append(entry['value'])
        
fitbit_steps = pd.DataFrame(data={
                'date':date,
                'steps': steps})
fitbit_steps['date'] = pd.to_datetime(fitbit_steps['date'])
fitbit_steps = fitbit_steps.set_index('date')

twitter_data['date'] = twitter_data.index.date
tweets_per_day = twitter_data['date'].value_counts()
twitter_mean = twitter_data.groupby(twitter_data.index.date).mean()
twitter_mean = twitter_mean.join(tweets_per_day)
twitter_mean['tweets_per_day'] = twitter_mean['date']

joined_data = twitter_mean.join(fitbit_steps,how='right')
joined_data['steps'] = joined_data['steps'].apply(int)
joined_data = joined_data[joined_data['steps'] != 0]
joined_data['year'] = joined_data.index.year

joined_data.head()

sns.lmplot(x="steps", 
           y="tweets_per_day", 
           data=joined_data[joined_data.index.year == 2014],
           lowess=True,
           size=8,
           aspect=1.5,
           scatter_kws={'alpha':0.3})

sns.lmplot(x="steps", y="tweets_per_day", row='year', data=joined_data,lowess=True,size=2,aspect=2,scatter_kws={'alpha':0.1})



