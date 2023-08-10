import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import LogNorm

from textblob.sentiments import NaiveBayesAnalyzer

import pandas as pd
import sqlite3
from textblob import TextBlob

import seaborn as sns
sns.set(color_codes=True)

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import spacy
nlp = spacy.load('en')

import re

df = pd.read_csv('../pitchfork3.csv')

years = np.arange(1999, 2018)
df_new = df[df['new_album'] == 1]
df_reissue = df[df['new_album'] == 0]

df_spotify = pd.read_csv('spotify_with_rank.csv')

df_merged = df.merge(df_spotify, how='left', on=['artist', 'title'])

df_merged.shape

sns.set_style("whitegrid")
sns.set_style('ticks')

filtered_df = df[
                  (df['score'] >= 9)  & \
                  (df['new_album'] == 1) & (df['num_prev_reviews'] == 1) & (df['pub_year'] < 2010 )]
filtered_df

plt.figure(figsize=(10,8))

plt.title('Average Score Where Score > 8.0')

x = years
y = df_merged[(df_merged['new_album'] == 1) & 
              (df_merged['score'] >= 8.0 )].groupby('pub_year')['score'].mean()
z = np.polyfit(x, y, 3)
p = np.poly1d(z)
plt.plot(x,p(x),"r--", lw=3)
plt.bar(x, y)
plt.ylim(8, 9)
plt.xticks(years, rotation=45)
sns.despine()
plt.show()

temp_df = df[(df['new_album'] == 1) & (df['score'] >= 8)]
temp_df['oft_reviewed'] = np.where(temp_df['num_prev_reviews'] > 1, "2+ Reviews", "First Review") 
plt.figure(figsize=(10,8))
plt.title('Distribution of Score for 8.0+ Reviews', fontsize=20)

sns.boxplot(x=temp_df['era'][::-1], y=temp_df['score'], 
            data=temp_df[['score', 'era']], 
            hue=temp_df['oft_reviewed'][::-1],
           notch=True)
plt.legend(fontsize=14, loc=1)
plt.tight_layout()

df_9 = df[(df['score'] >= 9)  & (df['best_new_music']==1) & (df['new_album'] == 1) & (df['num_prev_reviews'] ==1)][['artist', 'title', 'score', 'pub_year']]
df_9

filter_artists = ['anohni', 'jamie xx', 'dangelo, the vanguard', 'big boi', 'brian wilson', 'loretta lynn', 'ghostface killah']

df_9[~df_9['artist'].isin(filter_artists)]

sns.set_style('ticks')

# x: year
# y: score 8.0 - 10.0
# z: count of first/second releases with that score

filtered_df = df[ (~df['artist'].isin(filter_artists)) &                   (df['score'] >= 9)  &                   (df['new_album'] == 1)                     & (df['num_prev_reviews'] == 1)                     ]
temp_df = pd.DataFrame(filtered_df.groupby(['pub_year', 'score'])['reviewid'].count()).reset_index()
heatmap_df = temp_df.pivot(index='score', columns='pub_year', values='reviewid')

plt.figure(figsize=(14,9))
# plt.title("Artists Pitchfork has Reviewed More than Once")
# plt.suptitle('Count of Albums Receiving Scores +9.0')
plt.xlabel('Year')
plt.ylabel('Score')
sns.heatmap(heatmap_df[::-1], 
            annot=(heatmap_df[::-1]).round(3), 
            norm=LogNorm(vmin=0, vmax=1), 
            fmt='g', 
            linewidths=1,
            cbar=False,
            square=True,
            xticklabels=range(1999,2018),
            cmap=sns.light_palette('#2a3990', n_colors=16)
           )
plt.savefig("firsttimeartist9")









plt.figure(figsize=(10,8))

plt.title('Average Score Where Score > 8.0 -- First Time Reviewed by Pitchfork', fontsize=16)

x = years[4:19]
y = df_merged[(df_merged['best_new_music'] == 1) & (df_merged['new_album'] == 1) & 
              (df_merged['score'] >= 8.0) &
              (df_merged['order_of_release'] >= 0)].groupby('pub_year')['score'].mean()
y_1 = df_merged[(df_merged['best_new_music'] == 0) & (df_merged['new_album'] == 1) & 
              (df_merged['score'] >= 8.0) &
              (df_merged['order_of_release'] >= 0)].groupby('pub_year')['score'].mean()
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"r--")
plt.bar(x-.3, y, width=.6, label="Best New Music")
plt.bar(range(1999,2018), y_1, width=.6, label="Not Best New Music")
plt.legend(fontsize=16)
plt.xticks(years, rotation=45)
plt.ylim(8,9)
plt.show()



