from datetime import date
from statistics_aggregator import StatisticsAggregator
from data_aggregator import DataAggregator

data_helper = DataAggregator()
df = data_helper.get_data(date_range=['2017-08-16'])
stats_helper = StatisticsAggregator(df)
sdf = stats_helper.get_stats()

sdf

from data_enhancer import DataOrganizer
do = DataOrganizer(sdf)

ndf = do.enhance()

ndf

type(ndf['types'][0])

"dog".capitalize()

ndf = ndf.sort_values('created_at', ascending=False)

ndf

sdf.to_csv('stats.csv')
b = [dict((el, []) for el in ['cheese', 'fart', 'dog']), dict((el, []) for el in ['cheese', 'fart', 'dog'])]

import pandas as pd

csv_path = "stats.csv"
adf = pd.read_csv(csv_path)

type(adf['types'][0])

import ast

a = eval(adf['types'][0])



[(text,url) for text,url in zip(sdf.sort_values('hotness', ascending=False)['text'], sdf.sort_values('hotness', ascending=False)['url'])]

from twitter_statistics import TwitterStatistics
from data_aggregator import DataAggregator

data_helper = DataAggregator()
date_range = '2017-08-16' 
df = data_helper.get_data(date_range=date_range)

tweet_stats = TwitterStatistics(df)
a = tweet_stats.top_hashtags(5)
b = tweet_stats.top_mentions(3)

a = tweet_stats.top_hashtags(5)
b = tweet_stats.top_mentions(3)

print(a)
print(b)



