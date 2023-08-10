get_ipython().magic('matplotlib inline')

import glob
import os
import json
from pathlib import Path
from pandas import Series, DataFrame
import pandas as pd

p = Path(os.getcwd())
user_path = str(p.parent) + '/data/yelp_training_set/yelp_training_set_user.json'

df = pd.DataFrame(columns=['review', 'stars'])
with open(user_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        data = json.loads(line)
        review = data['review_count']
        stars = data['average_stars']
        df = df.append(Series({'review': review, 'stars': stars}), ignore_index=True)
df.head(5)

df['weight'] = df['review'] * df['stars']
total_reviews = df['review'].sum()
total_reviews

weighted_average = df['weight'].sum() / total_reviews
print(weighted_average)

df.head(5)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.style.use('seaborn-whitegrid')
df.plot(x='stars', y='review', kind = 'scatter', xlim = [0, 5], ylim = [0, 3000], color='Red', title = 'Reviews & Stars')
plt.savefig('ana_5/reviews_and_average_stars.png')

li = range(6)
filter_values = li[0::1]
out = pd.cut(df['stars'], bins = filter_values)
counts = pd.value_counts(out)
counts

cut_df = counts.to_frame().reset_index()
cut_df.columns = ['range_stars', 'count']
cut_df['stars'] = cut_df['range_stars'].apply(lambda x: x.split(', ')[1][:-1])
cut_df = cut_df.sort_values(by = 'stars')
cut_df = cut_df[['range_stars', 'count']]
cut_df.plot(x='range_stars', y='count', kind = 'bar', ylim = [0, 20000], title = 'Users in different star ranges')
plt.savefig('ana_5/users_in_star_range.png')

filter_values = [0, 10, 100, 500, 1000, 3000]
out_review = pd.cut(df['review'], bins = filter_values)
counts_review = pd.value_counts(out_review)
counts_review

cut_df = counts_review.to_frame().reset_index()
cut_df.columns = ['range_reviews', 'count']
cut_df['tmp'] = cut_df['range_reviews'].apply(lambda x: x.split(', ')[1][:-1])
cut_df = cut_df.sort_values(by = 'tmp')
cut_df = cut_df[['range_reviews', 'count']]
cut_df.plot(x='range_reviews', y='count', kind = 'bar', ylim = [0, 20000], title = 'Users in different review ranges')
plt.savefig('ana_5/users_in_review_range.png')

filter_values = [0, 3, 10, 100, 500, 1000, 3000]
new_df = df.groupby(pd.cut(df['review'], bins = filter_values)).mean()
new_df = new_df[['stars']].reset_index()
new_df

new_df.plot(x='review', y='stars', kind = 'line', ylim = [3.68, 3.8], title = 'User reviews and average stars')
plt.savefig('ana_5/users_stars_and_reviews.png')



