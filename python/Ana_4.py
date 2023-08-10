get_ipython().magic('matplotlib inline')

import glob
import os
import json
from pathlib import Path
from pandas import Series, DataFrame
import pandas as pd

p = Path(os.getcwd())
business_path = str(p.parent) + '/data/yelp_training_set/yelp_training_set_business.json'

import re

df = pd.DataFrame(columns=['city', 'street', 'type', 'count'])
play_keywords = ['Bars', 'Nightlife']
with open(business_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        data = json.loads(line)
        review_count = data['review_count']
        stars = data['stars']
        if review_count < 10 or stars < 4.0:
            continue
        address = data['full_address'].split('\n')
        if not re.match(r'^[0-9]', address[0]):
            continue
        city = address[-1].split(',')[0]
        street = ' '.join(address[0].split(' ')[1:])
        categories = data['categories']
        if 'Restaurants' in categories:
            df = df.append(Series({'city': city, 'street': street, 'type': 'restaurant', 'count': 1}), ignore_index=True)
        if not set(play_keywords).isdisjoint(categories):
            df = df.append(Series({'city': city, 'street': street, 'type': 'nightlife', 'count': 1}), ignore_index=True)
        if 'Restaurants' not in categories and set(play_keywords).isdisjoint(categories):
            df = df.append(Series({'city': city, 'street': street, 'type': 'other', 'count': 1}), ignore_index=True)
df.head(5)

total_df = df.groupby(['city', 'street']).sum().reset_index()
total_df.head(5)

profitable = total_df.groupby(['city']).max().reset_index()
profitable.head(5)

profitable.to_csv('ana_4/profitable_streets_in_AZ.csv', index = False, header = True)

restaurant_df = df[df['type'] == 'restaurant']
restaurant_df = restaurant_df.groupby(['city', 'street']).sum().reset_index().groupby(['city']).max().reset_index()
restaurant_df.head(5)

restaurant_df.to_csv('ana_4/restaurant_streets_in_AZ.csv', index = False, header = True)

nightlife_df = df[df['type'] == 'nightlife']
nightlife_df = nightlife_df.groupby(['city', 'street']).sum().reset_index().groupby(['city']).max().reset_index()
nightlife_df.to_csv('ana_4/nightlife_streets_in_AZ.csv', index = False, header = True)
nightlife_df.head(5)



