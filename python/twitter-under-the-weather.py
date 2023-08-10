DARKSKY_KEY = 'ENTER_YOUR_KEY_HERE'
DATARANGE_START = "2016-06-01"
DATARANGE_END = "2018-05-08"

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

weather_cache = {}

def longest_daily_location(daily_segments):
    """
    takes a daily segment log of Moves and returns the 
    lat/long for the location where most time was spent. 
    Can be misleading for days w/ lots of travel etc. 
    But the most quick/dirty solution for now.
    """
    places_of_day = []
    for i in daily_segments:
        if i['type'] == 'place':
            place_location = i['place']['location']
            start_time = datetime.strptime(i['startTime'],'%Y%m%dT%H%M%S%z')
            end_time = datetime.strptime(i['endTime'],'%Y%m%dT%H%M%S%z')
            duration = end_time - start_time
            places_of_day.append([place_location,duration])
    places_of_day.sort(key=lambda tup: tup[-1],reverse=True)
    return places_of_day[0][0]

def weather_for_location_day(location, date):
    """
    take a location as returned by longest_daily_location 
    and a date of YYYYMMDD to get the weather from the 
    Dark Sky API. Does not do any filtering yet
    """
    if DARKSKY_KEY == 'ENTER_YOUR_KEY_HERE':
        raise ValueError("Oops, you didn't add your Dark Sky API key! "
              "Scroll back up to the first step in this notebook. :)")
    date = '{}-{}-{}T12:00:00'.format(
    date[:4],
    date[4:6],
    date[6:8])

    if date not in weather_cache:
        url = 'https://api.darksky.net/forecast/{}/{},{},{}?units=si'.format(DARKSKY_KEY,
            location['lat'],
            location['lon'],
            date
        )
        response = requests.get(url)
        weather_cache[date] = response.json()

    return weather_cache[date]

user = api.exchange_oauth2_member(os.environ.get('OH_ACCESS_TOKEN'))
has_twitter = False
has_moves = False

# get our download URLs
for entry in user['data']:
    if entry['source'] == "direct-sharing-70":
        twitter_data_url = entry['download_url']
        has_twitter = True
    if entry['source'] == "direct-sharing-138":
        moves_data_url = entry['download_url']
        has_moves = True

if not has_twitter:
    print("YOU NEED TO HAVE SOME TWITTER DATA IN YOUR ACCOUNT TO USE THIS NOTEBOOK")
    print("GO TO http://twarxiv.org TO UPLOAD IT")

if not has_twitter:
    print("YOU NEED TO HAVE SOME MOVES DATA IN YOUR ACCOUNT TO USE THIS NOTEBOOK")
    print("GO TO https://ohmovessource.herokuapp.com/ TO UPLOAD IT")
    
# read the twitter data
twitter_data = create_main_dataframe(zip_url=twitter_data_url)

# read the moves data
moves_data_raw = requests.get(moves_data_url).content
moves_data = json.loads(moves_data_raw)

twitter_data.head()


weather_variables  = ['precipType','precipIntensity','temperatureHigh',
                      'temperatureLow','precipIntensityMax','precipAccumulation',
                      'apparentTemperatureHigh', 'apparentTemperatureLow']

moves_processed_data = defaultdict(list)

for datapoint in moves_data:
    # we need to have observed segments for that day. If moves wasn't running we ignore the day
    if datapoint['segments'] != None:
        # did we stay in a place that day and did we walk that day?
        has_places = False
        walked = False
        for i in datapoint['segments']:
            if i['type'] == 'place':
                    # yes, we were in one place w/o moving around too much, we can keep this day
                    has_places = True
        # is this day in our date range of interest and has data?
        if datapoint['summary'] != None and has_places and datetime.strptime(datapoint['date'],"%Y%m%d") > datetime.strptime(DATARANGE_START,"%Y-%m-%d"):
            moves_processed_data['date'].append(datapoint['date'])
            for activity in datapoint['summary']:
                if activity['activity'] == 'walking':
                    moves_processed_data['steps'].append(activity['steps'])
                    moves_processed_data['distance'].append(activity['distance'])
                    walked = True
            # whops, we have moves data for this day but moves didn't register you walking & your step count is zero
            if not walked:
                moves_processed_data['steps'].append(0)
                moves_processed_data['distance'].append(0)    
            # grab the location where we spent most time as baseline geo location for the day
            location = longest_daily_location(datapoint['segments'])
            moves_processed_data['lat'].append(location['lat'])
            moves_processed_data['lon'].append(location['lon'])
            # now we can grab the weather for our baseline location
            weather = weather_for_location_day(location,datapoint['date'])
            for variable in weather_variables:
                if variable in weather['daily']['data'][0].keys():
                    moves_processed_data[variable].append(weather['daily']['data'][0][variable])
                else:
                    moves_processed_data[variable].append(0)
            if datetime.strptime(datapoint['date'],"%Y%m%d") > datetime.strptime(DATARANGE_END,"%Y-%m-%d"):
                break

moves_dataframe = pd.DataFrame(data={
    'date': moves_processed_data['date'],
    'steps': moves_processed_data['steps'],
    'distance': moves_processed_data['distance'],
    'latitude': moves_processed_data['lat'],
    'longitude': moves_processed_data['lon'],
    'precipType': moves_processed_data['precipType'],
    'precipIntensity': moves_processed_data['precipIntensity'],
    'precipIntensityMax': moves_processed_data['precipIntensityMax'],
    'precipAccumulation': moves_processed_data['precipAccumulation'],
    'temperatureHigh': moves_processed_data['temperatureHigh'],
    'temperatureLow': moves_processed_data['temperatureLow'],
    'apparentTemperatureHigh': moves_processed_data['apparentTemperatureHigh'],
    'apparentTemperatureLow': moves_processed_data['apparentTemperatureLow']

})

moves_dataframe.head()

polarity = []
subjectivity = []
twitter_data['blob'] = twitter_data['text'].apply(TextBlob)
for entry in twitter_data['blob']:
    polarity.append(entry.sentiment.polarity)
    subjectivity.append(entry.sentiment.subjectivity)
twitter_data['polarity'] = polarity
twitter_data['subjectivity'] = subjectivity
twitter_data = twitter_data.replace(0, np.nan)

twitter_data['date'] = twitter_data.index.date
tweets_per_day = twitter_data['date'].value_counts()
twitter_mean = twitter_data.groupby(twitter_data.index.date).mean()
twitter_mean = twitter_mean.join(tweets_per_day)

twitter_mean.head()

moves_dataframe['date'] = moves_dataframe['date'].apply(lambda x: datetime.strptime(x,"%Y%m%d"))
moves_dataframe.index = moves_dataframe['date']
twitter_moves = twitter_mean.join(moves_dataframe,lsuffix='_twitter',rsuffix='_moves',how='right')

sns.lmplot(x="apparentTemperatureHigh", y="polarity", data=twitter_moves,lowess=True,size=8,aspect=1.5)

sns.lmplot(x="apparentTemperatureHigh", y="subjectivity", data=twitter_moves,lowess=True,size=8,aspect=1.5)

sns.violinplot(x='precipType',
               y='polarity',
               data=twitter_moves[twitter_moves['precipType'] != 'snow'])

sns.violinplot(x='precipType',
               y='subjectivity',
               data=twitter_moves[twitter_moves['precipType'] != 'snow'])

from scipy.stats import mannwhitneyu

twitter_moves

rain = twitter_moves[twitter_moves['precipType'] == 'rain']
clear = twitter_moves[twitter_moves['precipType'] == 0]
polarity_results = mannwhitneyu(rain['polarity'], clear['polarity'],alternative='two-sided')
subjectivity_results  = mannwhitneyu(rain['subjectivity'], clear['subjectivity'],alternative='two-sided')
print('the p-value for a subjectivity-difference between rain & shine is {}'.format(subjectivity_results.pvalue))
print('--')
print('the p-value for a polarity-difference between rain & shine is {}'.format(polarity_results.pvalue))

tweet_year = "2016"
# you can get these values from Google Maps by clicking on a point and copy&pasting the values 
# from the URL/bottom information box

target_latitude = 50.114462 # If you're south of the equator this one should start with "-"
target_longitude = 8.746019 # If you're west of the prime meridian this should also start with a "-"
km_distance = 25 # we don't want to be too strict in filtering. 

# earth's radius in km = ~6371
earth_radius = 6371

# latitude boundaries
maxlat = target_latitude + np.rad2deg(km_distance / earth_radius)
minlat = target_latitude - np.rad2deg(km_distance / earth_radius)

#longitude boundaries (longitude gets smaller when latitude increases)
maxlng = target_longitude + np.rad2deg(km_distance / earth_radius / np.cos(np.deg2rad(target_latitude)))
minlng = target_longitude - np.rad2deg(km_distance / earth_radius / np.cos(np.deg2rad(target_latitude)))

twitter_moves['tweets_per_day'] = twitter_moves['date_twitter'] 
twitter_moves_subset = twitter_moves[(twitter_moves.index.year == int(tweet_year)) & 
                                    (twitter_moves['longitude_moves'] >= minlng) & 
                                    (twitter_moves['longitude_moves'] < maxlng) &
                                    (twitter_moves['latitude_moves'] < maxlat) & 
                                    (twitter_moves['latitude_moves'] > minlat)]


sns.lmplot(x="temperatureLow", y="tweets_per_day", data=twitter_moves_subset,lowess=True,size=8,aspect=1.5)

sns.violinplot(x='precipType',
               y='tweets_per_day',
               data=twitter_moves_subset[twitter_moves_subset['precipType'] != 'snow'])

