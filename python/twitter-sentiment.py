# LOAD OUR MODULES 
get_ipython().run_line_magic('matplotlib', 'inline')

import os, requests, json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import zipfile
import pytz
import io
import sys
from textblob import TextBlob
import emoji
from ohapi import api

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
        utc_time.append(datetime.datetime.strptime(single_tweet['created_at'],
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
for entry in user['data']:
    if entry['source'] == "direct-sharing-70":
        twitter_data_url = entry['download_url']
        break
twitter_data_url

twitter_data = create_main_dataframe(zip_url=twitter_data_url)

twitter_data.head()

polarity = []
subjectivity = []
twitter_data['blob'] = twitter_data['text'].apply(TextBlob)
for entry in twitter_data['blob']:
    polarity.append(entry.sentiment.polarity)
    subjectivity.append(entry.sentiment.subjectivity)
twitter_data['polarity'] = polarity
twitter_data['subjectivity'] = subjectivity
twitter_data = twitter_data.replace(0, np.nan)

# get 30-day averaged standard deviations for polarity & subjectivity
twitter_std = twitter_data.groupby(twitter_data.index.date).std()
twitter_std.index = pd.to_datetime(twitter_std.index)
twitter_std_rolling = twitter_std.rolling('30d').mean()

# get 30-day mean averaged daily means for polarity & subjectivity
twitter_mean = twitter_data.groupby(twitter_data.index.date).mean()
twitter_mean.index = pd.to_datetime(twitter_mean.index)
twitter_mean_rolling = twitter_mean.rolling('30d').mean()

# get 30-day averaged maximum values for polarity & subjectivity
twitter_max = twitter_data.groupby(twitter_data.index.date).max()
twitter_max.index = pd.to_datetime(twitter_max.index)
twitter_max_rolling = twitter_max.rolling('30d').mean()

# get 30-day averaged minimum for polarity & subjectivity
twitter_min = twitter_data.groupby(twitter_data.index.date).min()
twitter_min.index = pd.to_datetime(twitter_min.index)
twitter_min_rolling = twitter_min.rolling('30d').mean()

polarity = pd.DataFrame(data={
    "max_polarity": twitter_max_rolling["polarity"],
    "mean_polarity": twitter_mean_rolling["polarity"],
    "min_polarity": twitter_min_rolling["polarity"],    
    "std_polarity": twitter_std_rolling["polarity"]    
})

subjectivity = pd.DataFrame(data={
    "max_subjectivity": twitter_max_rolling["subjectivity"],
    "mean_subjectivity": twitter_mean_rolling["subjectivity"],
    "min_subjectivity": twitter_min_rolling["subjectivity"],    
    "std_subjectivity": twitter_std_rolling["subjectivity"]    
})

pt = polarity.plot(y=['max_polarity','mean_polarity','min_polarity','std_polarity'],figsize=(15,10),fontsize=14)
pt.legend(['Maximum Polarity','Mean Polarity','Minium Polarity','Standard Deviation of Polarity'])

pt = subjectivity.plot(y=['max_subjectivity','mean_subjectivity','min_subjectivity','std_subjectivity'],figsize=(15,10),fontsize=14)
pt.legend(['Maximum subjectivity','Mean subjectivity','Minimum subjectivity','Standard Deviation of subjectivity'])

def number_emoji(row):
    n_emoji = 0
    for character in row['text']:
        if character in emoji.UNICODE_EMOJI:
            n_emoji += 1
    return n_emoji

twitter_data['emoji_count'] = twitter_data.apply(number_emoji, axis=1)

twitter_emoji = twitter_data.groupby(twitter_data.index.date).sum()
twitter_emoji.index = pd.to_datetime(twitter_emoji.index)
twitter_emoji_rolling = twitter_emoji.rolling('90d').mean()

pt = twitter_emoji_rolling.plot(y=['emoji_count'],figsize=(15,10),fontsize=14)
pt.legend(['daily emoji count'])

from collections import defaultdict

emojis = defaultdict(int)
for tweet in twitter_data['text']:
    for character in tweet:
        if character in emoji.UNICODE_EMOJI:
            emojis[character] += 1

s = [(k, emojis[k]) for k in sorted(emojis, key=emojis.get, reverse=True)]
for k,v in s:
    print(k,v)
    if v < 5:
        break

emoji_love = ["💖","😍","😘","🍆","🌈","💘","💓","💕","♥","💜"]
emoji_science = ['📉','🌱','🐌','🤓','🔬','📚','📊','🍄']
emoji_joy = ['😋','😛','😜','😀','😎','☺','😊','😉','👍','😂']
emoji_celebrate = ['☑','🍺','✨','🍾','🎈','🎊','✅','🍻','✔','🎉']
emoji_sad = ['😕','🤦','😥','😳','😒','😞','😔','💩','🔥','🤷','😭','😢','😱']
emoji_travel = ['🌍','🚙','🚂','🚗','✈']

def classify_number_emoji(row, classifier):
    n_emoji = 0
    for character in row:
        if character in classifier:
            n_emoji += 1
    return n_emoji

twitter_data['emoji_count_love'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_love,))
twitter_data['emoji_count_science'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_science,))
twitter_data['emoji_count_joy'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_joy,))
twitter_data['emoji_count_celebrate'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_celebrate,))
twitter_data['emoji_count_sad'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_sad,))
twitter_data['emoji_count_travel'] = twitter_data['text'].apply(classify_number_emoji,args=(emoji_travel,))

twitter_emoji = twitter_data.groupby(twitter_data.index.date).sum()
twitter_emoji.index = pd.to_datetime(twitter_emoji.index)
twitter_emoji_rolling = twitter_emoji.rolling('90d').mean()

pt = twitter_emoji_rolling.plot(y=['emoji_count_love',
                                   #'emoji_count_science',
                                   'emoji_count_joy',
                                   'emoji_count_celebrate',
                                   'emoji_count_sad'],
                                   #'emoji_count_travel'],
                                figsize=(15,10),
                                fontsize=14,
                                xlim=["2015-01-01","2018-03-01"])
pt.legend(['emoji_count_love',
        #'emoji_count_science',
        'emoji_count_joy',
        'emoji_count_celebrate',
        'emoji_count_sad',])
        #'emoji_count_travel'])



