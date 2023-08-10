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

df['score'].mean()

df = pd.read_csv('../pitchfork3.csv')

df['word_count'] = df['content'].str.count('\w+')
df['adj_count'] = df['content_adj'].str.count('\w+')

df['avg_word_length'] = df['content'].str.count('[^ ]') / df['word_count']
df['adj_word_length'] = df['content_adj'].str.count('[^ ]') / df['content_adj'].str.count('\w+')
df['adj_freq'] = df['adj_count'] / df['word_count']

df['best_new_reissue'].sum()

# This loop adds data for an artist's previous reviews on Pitchfork

prev_data = []
for i in range(len(df['score'])):
    reviewid = df['reviewid'][i]
    artist = df['artist'][i]
    sub_df = df[(df['pub_date'] < df['pub_date'][i]) & (df['artist'] == df['artist'][i])]
    if df['artist'][i] == 'various artists':
        num_reviews = 1
        num_bnm = 0
    else:
        num_reviews = sub_df['score'].count() + 1
        num_bnm = sub_df['best_new_music'].sum()
    if num_reviews == 1:
        avg_scores = df['score'][i]
    else:
        avg_scores = sub_df['score'].mean()
    
    prev_data.append((reviewid, artist, num_reviews, avg_scores, num_bnm))

df_prev = pd.DataFrame(prev_data, columns =['reviewid', 'artist', 'num_prev_reviews', 'avg_score_of_prev_reviews', 'number_prev_bnm'])

df = df.merge(df_prev)

values_dict = {1999: '1: Pre BNM', 
               2000: '1: Pre BNM', 
               2001: '1: Pre BNM', 
               2002: '1: Pre BNM', 
               2003: '2: Pre PMF',
               2004: '2: Pre PMF', 
              2005:  '2: Pre PMF', 
              2006:  '3: Pre Redesign', 
              2007:  '3: Pre Redesign', 
              2008:  '3: Pre Redesign', 
              2009:  '3: Pre Redesign', 
              2010:  '3: Pre Redesign', 
              2011:  '4: Pre CD', 
              2012:  '4: Pre CD', 
              2013:  '4: Pre CD', 
               2014:  '4: Pre CD', 
               2015:  '4: Pre CD', 
               2016:  '5: Current Era', 
               2017:  '5: Current Era', 
              }

df['era'] = df['pub_year'].map(values_dict)

df.describe()

df.to_csv('../pitchfork2.csv')

df = pd.read_csv('../pitchfork2.csv')

years = np.arange(1999, 2018)
df_new = df[df['new_album'] == 1]
df_reissue = df[df['new_album'] == 0]

df_spotify = pd.read_csv('spotify_with_rank.csv')

df_merged = df.merge(df_spotify, how='left', on=['artist', 'title'])

#df.drop('abstract_adj', axis=1, inplace=True)

df['abstract'].fillna(' ', inplace=True)

plt.title('Count of Total Reviews by Year')
plt.plot(df.groupby('pub_year').count()['reviewid'])
plt.xticks(range(1999,2018), rotation=45)
plt.show()

plt.plot(df.groupby('pub_year').sum()['reissue'])
plt.title("Count of Reissue/Classic Reviews Per Year")
plt.xticks(range(1999,2018), rotation=45)
plt.show()

plt.plot(df.groupby('pub_year').sum()['best_new_music'])
plt.title("Count of Best New Music Reviews Per Year")
plt.xticks(range(1999,2018), rotation=45)
plt.show()

plt.hist(df_new[df_new['best_new_music'] == 1]['score'], bins=10)
plt.title('Count of Best New Music Albums')
plt.xticks(np.arange(5,10.5,.5))
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()

plt.hist(df[df['reissue'] == 1]['score'], bins=10)
plt.title('Count of Best New Reissue Albums')
# plt.xticks(np.arange(5,10.5,.5))
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()

df[df['num_prev_reviews'] == 1]['reviewid'].count()/float(df['reviewid'].count())

plt.plot(years, (df[df['num_prev_reviews'] == 1].groupby('pub_year')['reviewid'].count()/df.groupby('pub_year')['reviewid'].count()))
plt.title('Proportion of Reviews by Previously Unreviewed Artist')
plt.xticks(years, rotation=45)
plt.xlabel('% of Reviews')
plt.ylim(0,1)
plt.show()

filtered_df = df_new
temp_df = pd.DataFrame(filtered_df.groupby(['pub_year', 'num_prev_reviews'])['reviewid'].count()).reset_index()
heatmap_df = temp_df.pivot(index='num_prev_reviews', columns='pub_year', values='reviewid')

plt.figure(figsize=(16,8))
plt.title("Count of Reviews Based on # of Previous REVIEWS")
plt.xlabel('Year')
plt.ylabel('Score')
sns.heatmap(heatmap_df[::-1], 
            annot=(heatmap_df[::-1]).round(3), 
            norm=LogNorm(vmin=0, vmax=1), 
            fmt='g', 
            cmap=sns.light_palette((250, 80, 40), input='husl', n_colors=16)
           )
plt.show()

df_merged[df_merged['new_album'] == 1].groupby('pub_year').mean()['score']

plt.plot(years, df_merged[(df_merged['new_album'] == 1) & (df_merged['order_of_release'] <= 2)].groupby('pub_year').mean()['score'])
plt.plot(years, df_merged[(df_merged['new_album'] == 1)].groupby('pub_year').mean()['score'])
plt.xticks(years, rotation=45)
plt.ylim(6,8)
plt.title('Avg Score of First-Time-Reviewed Artists')
plt.show()

df_merged['binned_release'] = pd.cut(df_merged['order_of_release'], bins=[0,2.1,5.1,10,50] )

filtered_df = df_merged
temp_df = pd.DataFrame(filtered_df.groupby(['pub_year', 'binned_release'])['score'].mean()).reset_index()
heatmap_df = temp_df.pivot(index='binned_release', columns='pub_year', values='score')

plt.figure(figsize=(16,8))
plt.title("Count of Reviews Based on # of Previous REVIEWS")
plt.xlabel('Year')
plt.ylabel('Score')
plt.ylim(0,20)
sns.heatmap(heatmap_df[::-1], 
            annot=(heatmap_df[::-1]).round(3), 
#             norm=LogNorm(vmin=0, vmax=1), 
            fmt='g', 
            cmap=sns.light_palette((250, 80, 40), input='husl', n_colors=11)
           )
plt.show()

plt.title("Review Distribution - New Releases")
sns.distplot(df_new['score'], bins=30, kde=False)
plt.xlim(0,10)

plt.show()

plt.title("Review Distribution - Reissues/Classics")
sns.distplot(df_reissue['score'], bins=30, kde=False)
plt.xlim(0,10)
plt.show()

plt.plot(df_new['pub_year'].unique(), df_new.groupby('pub_year')['score'].mean(), label='New Releases')
plt.plot(df_reissue['pub_year'].unique(), df_reissue.groupby('pub_year')['score'].mean(), label='Reissues / Classic Albums')
plt.legend()
plt.title('Mean Score by Year - New Releases')
plt.ylim(0,10)
plt.xticks(range(1999, 2018), rotation=45)
plt.show()

less3 = list(df_new[df_new['score'] <= 3].groupby('pub_year').count()['reviewid'] / (df_new.groupby('pub_year')['reviewid'].count().astype(float)))
less6 = list(df_new[df_new['score'] <= 6].groupby('pub_year').count()['reviewid'] / (df_new.groupby('pub_year')['reviewid'].count().astype(float)))
more9 = list(df_new[df_new['score'] >= 9].groupby('pub_year').count()['reviewid'] / (df_new.groupby('pub_year')['reviewid'].count().astype(float)))
reis_more9 = list(df_reissue[df_reissue['score'] >= 9].groupby('pub_year').count()['reviewid'] / (df_reissue.groupby('pub_year')['reviewid'].count().astype(float)))

plt.bar(years, less3)
plt.xticks(years, range(1999, 2018),rotation =45)
plt.title("Percentage of Reviews Scoring < 3")
plt.show()

plt.bar(years, less6)
plt.xticks(years, range(1999, 2018),rotation =45)
plt.title("Percentage of Reviews Scoring < 6")

plt.show()

plt.bar(years, more9)
plt.xticks(years, range(1999, 2018),rotation =45)
plt.title("Percentage of New Release Reviews Scoring Above 9")

plt.show()

fig, ax = plt.subplots(figsize=(12,8))
plt.title('Distribution of Scores by Period - New Releases')
sns.distplot(df_new[(df_new['pub_year'] > 2014) & (df_new['pub_year'] < 2018)]['score'], label='2016-2017', bins=20)
sns.kdeplot(df_new[(df_new['pub_year'] > 2011) & (df_new['pub_year'] < 2015)]['score'], shade=True, label='2012-2015')
sns.kdeplot(df_new[(df_new['pub_year'] > 2007) & (df_new['pub_year'] < 2012)]['score'], shade=True, label='2007-2011')
sns.kdeplot(df_new[df_new['pub_year'] < 2007]['score'], shade=True, label='1999-2006')
plt.xlabel('score')
plt.xticks(range(0,11))
plt.xlim(0,10)
plt.ylabel('% captured')
plt.legend()
plt.show()

temp_df = pd.DataFrame(df_new.groupby(['era', 'score'])['reviewid'].count()).reset_index()
heatmap_df = temp_df.pivot(index='score', columns='era', values='reviewid').fillna(.001)
df_norm_col=(heatmap_df)/heatmap_df.sum()

plt.figure(figsize=(12,8))
plt.title("Score % Distribution by Year")
plt.xlabel('Year')

plt.ylabel('Score')
sns.heatmap(df_norm_col[::-1], 
#             annot=(df_norm_col[::-1]).round(3), 
            norm=LogNorm(vmin=0, vmax=1), 
            fmt='.1%', 
            cmap=sns.light_palette((250, 70, 40), input='husl', n_colors=12)
           )
plt.show()

temp_df = pd.DataFrame(df_new.groupby(['pub_year', 'score'])['reviewid'].count()).reset_index()
heatmap_df = temp_df.pivot(index='score', columns='pub_year', values='reviewid').fillna(.001)
df_norm_col=(heatmap_df)/heatmap_df.sum()

plt.figure(figsize=(14,12))
plt.title("Score % Distribution by Year")
plt.xlabel('Year')

plt.ylabel('Score')
sns.heatmap(df_norm_col[::-1], 
#             annot=(df_norm_col[::-1]).round(3), 
            norm=LogNorm(vmin=0, vmax=1), 
            fmt='.1%', 
            cmap=sns.light_palette((250, 70, 40), input='husl', n_colors=12)
           )
plt.show()

plt.figure(figsize=(12,8))

sns.boxplot(x=df_new['pub_year'], y=df_new['score'], data=df_new[['score', 'pub_year']])
plt.title('Score Distribution by Year - New Releases')
plt.yticks(np.arange(0,11))

plt.xticks(rotation=45)
plt.show()

score_by_year = df_new.groupby('pub_year').agg(['mean', 'max','min','count','sum']).reset_index()

score_by_year['percent_bnm'] = score_by_year['best_new_music']['sum'] / score_by_year['score']['count']

plt.plot(score_by_year['pub_year'][4:19], score_by_year['percent_bnm'][4:19])
plt.title('Best New Music Frequency')
plt.xticks(score_by_year['pub_year'][4:19], rotation=45)
plt.yticks()
plt.xlabel('Year')
plt.ylabel('% of Albums Named Best New Music')
plt.show()

# box plot of avg word count

plt.figure(figsize=(12,8))

sns.boxplot(x=df['pub_year'], y=df['word_count'], data=df[['word_count', 'pub_year']])
plt.title('Word Count Distribution by Year - New Releases')
# plt.yticks(np.arange(0,11))

plt.xticks(rotation=45)
plt.show()

plt.plot(df['pub_year'].unique()[::-1], df.groupby('pub_year')['word_count'].mean())

plt.xticks(years, rotation=45)
plt.xlim(1999, 2018)
plt.xlabel('Year')
plt.ylabel('Word Count')
plt.title('Avg Word Count by Year - All')
plt.show()

plt.plot(df_new['pub_year'].unique()[::-1], df_new.groupby('pub_year')['avg_word_length'].mean())

plt.xticks(years, rotation=45)
plt.xlabel('Year')
plt.ylabel('Word Count')
plt.title('Avg Word Length - New Releases')
plt.show()

plt.plot(df_new['pub_year'].unique()[::-1], df_new.groupby('pub_year')['adj_word_length'].mean())

plt.xticks(years, rotation=45)
plt.xlabel('Year')
plt.ylim(5, 7)
plt.ylabel('Word Count')
plt.title('Avg Word Length of Adjectives - New Releases')
plt.show()

plt.plot(df_new['pub_year'].unique()[::-1], df_new.groupby('pub_year')['adj_freq'].mean())

plt.xticks(years, rotation=45)
plt.xlabel('Year')
plt.ylabel('Adjective/Adverb Frequency')
plt.title('Adjective/Adverb Frequency by Year')
plt.show()

plt.plot(df_new['pub_year'].unique()[::-1], df_new.groupby('pub_year')['adj_freq'].mean())

plt.xticks(years, rotation=45)
plt.xlabel('Year')
plt.ylabel('Adjective/Adverb Frequency')
plt.title('Adjective/Adverb Frequency by Year')
plt.show()

# report on shortcomings of TextBlob dictionary

temp_df.groupby('pub_year')['cont_subjectivity'].mean()

temp_df = df[df['best_new_music'] == 1]
plt.plot(temp_df['pub_year'].unique()[::-1], temp_df.groupby('pub_year')['cont_subjectivity'].mean(), label='Content Subjectivity')
plt.plot(temp_df['pub_year'].unique()[::-1], temp_df.groupby('pub_year')['abs_subjectivity'].mean(), label='Abstract Subjectivity')

plt.xticks(years, rotation=45)
plt.xlim(2005, 2018)
plt.xlabel('Year')
plt.ylabel('Subjectivity')
plt.title('Avg Subjectivity by Year')
plt.legend()
plt.show()

plt.plot(df['pub_year'].unique()[::-1], df.groupby('pub_year')['cont_subjectivity'].mean(), label='Content Subjectivity')
plt.plot(df['pub_year'].unique()[::-1], df.groupby('pub_year')['abs_subjectivity'].mean(), label='Abstract Subjectivity')

plt.xticks(years, rotation=45)
plt.xlim(2005, 2018)
plt.xlabel('Year')
plt.ylabel('Subjectivity')
plt.title('Avg Subjectivity by Year')
plt.legend()
plt.show()

df_bnm = df_new[df_new['best_new_music'] == 1]

plt.plot(df_bnm['pub_year'].unique()[::-1], df_bnm.groupby('pub_year')['cont_polarity'].mean(), label='Content Polarity')
plt.plot(df_bnm['pub_year'].unique()[::-1], df_bnm.groupby('pub_year')['abs_polarity'].mean(), label='Abstract Polarity')

plt.xticks(years, rotation=45)
plt.xlim(2005, 2018)
plt.xlabel('Year')
plt.ylabel('Subjectivity')
plt.title('BNM Avg Polarity by Year')
plt.legend()
plt.show()

df_new[df_new['cont_polarity'] == -1]['non_PN_words']

plt.scatter(df_new['word_count'], df_new['score'])

plt.scatter(df_new['avg_word_length'], df_new['score'])

plt.scatter(df_new['adj_freq'], df_new['score'])

plt.scatter(df_new['abs_subjectivity'], df_new['score'], alpha=.3)
plt.title('Subjectivity Vs. Score')
plt.ylabel("Score")
plt.xlim(.25, .7)
plt.xlabel("Subjectivity")
plt.show()

df.head()

plt.scatter(df_new[df_new['pub_year'] > 2004]['abs_polarity'], df_new[df_new['pub_year'] > 2004]['score'], alpha=.3)
plt.title('Polarity Vs. Score')
plt.ylabel("Score")
plt.xlabel("Polarity")
plt.show()

import statsmodels.api as sm

df.fillna(0, inplace=True)

df.columns

df.head()

Y = df['score']
X = df[['avg_score_of_prev_reviews',
#         'cont_polarity', 
#         'avg_word_length',
        
#         'new_album',
#         'genre_electronic', 
#         'genre_experimental',
#         'genre_global', 
#         'genre_jazz',
#         'genre_pop/r&b', 
#         'genre_rap', 
#         'genre_rock', 
#         'cont_subjectivity',
#         'adj_freq', 
       ]]
X = sm.add_constant(X)

model = sm.OLS(Y,X)
results = model.fit()
results.summary()



Y = df['score']
X = df[['abs_polarity']]
X = sm.add_constant(X)

model = sm.OLS(Y,X)
results = model.fit()
results.summary()



def plot_subj_violins_score(df, genre='all genres'):
    scores = list(df.score_bin.unique())
    pol_dist = {1: [],}
    for pol,score in zip(df.subjectivity.astype(list), df.score_bin.astype(list)):
        if score not in pol_dist.keys():
            pol_dist[score] = []
        else:
            pol_dist[score] = pol_dist[score] + [pol]

    plt.figure(figsize=(30,14))
#     plt.hlines(df['subjectivity'].mean(), 0, 10, linestyles='dotted')
    for each in scores:
        plt.xticks(range(0,11), fontsize=16)
        plt.violinplot(np.array(pol_dist[each]), [each], widths=1,showmedians=True,showextrema=True)
    plt.title('Subjectivity Distribution by Score ({})'.format(genre), fontsize=24)
    plt.ylabel('Subjectivity', fontsize=16)
    plt.xlabel('Binned Score', fontsize=16)

sns.boxplot(x=df[df['best_new_music'] == 1]['pub_year'], y=df[df['best_new_music'] == 1]['cont_subjectivity'], data=df[df['best_new_music'] == 1][['score', 'cont_subjectivity']])
# plt.ylim(.28, .7)
plt.title("Subjectivity Distribution by Year -- Best New Music Albums")
plt.xticks(rotation=45)
# plt.yticks(np.arange(.3,.7,.025))

plt.show()

sns.boxplot(x=df['pub_year'], y=df['cont_subjectivity'], data=df[['score', 'cont_subjectivity']])
# plt.ylim(.28, .7)
plt.title("Subjectivity Distribution by Year -- All Albums")
# plt.yticks(np.arange(.3,.7,.025))

plt.xticks(rotation=45)
plt.show()

sns.boxplot(x=df_new['pub_year'], y=df_new['cont_subjectivity'], data=df_new[['score', 'cont_subjectivity']])
plt.title("Subjectivity Distribution by Year -- New Releases")
# plt.ylim(.28, .7)
plt.xticks(rotation=45)
# plt.yticks(np.arange(.3,.7,.025))

plt.show()

sns.boxplot(x=df_reissue['pub_year'], y=df_reissue['cont_subjectivity'], data=df_reissue[['score', 'cont_subjectivity']])
plt.title("Subjectivity Distribution by Year -- Reissues and Classic Albums")
plt.xticks(rotation=45)
# plt.yticks(np.arange(.3,.7,.025))

plt.show()

pd.DataFrame(df['artist'].unique()).to_csv('artists.csv')

pd.to_csv(len(list(df['artist'].unique())))

import pandas as pd

df = pd.read_csv('artists.csv')

df['artist'] = df['0']

list(df['artist'].unique()).index('the coup'
                                 )

df_spotify[df_spotify['artist'] == 'wilco']

df_merged.drop('abstract', axis=1, inplace=True)

df_not_on_spotify = df_merged[pd.isnull(df_merged).any(axis=1)]

df_merged[df_merged['artist'] == 'wilco']

df_merged = df.merge(df_spotify, on=['artist', 'title'])

df_merged.columns

df_new_artist_bnm = df_merged[(df_merged['best_new_music'] == 1) & (df_merged['order_of_release'] < 3) & (df_merged['new_album'] == 1)][['title', 'artist', 'score', 'pub_year', 'order_of_release']]

df_merged.shape

plt.figure(figsize=(10,8))
x = range(2003,2018)
y = df_new_artist_bnm.groupby('pub_year')['title']          .count()/df_merged[(df_merged['best_new_music'] == 1) &          (df_merged['new_album'] == 1)].groupby('pub_year')['reviewid'].count()
plt.plot(x, y)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")
plt.title("Percent of Best New Music Albums Where The Album is \n the Artist's First or Second Release")
plt.xticks(years[4:19], rotation=45)
plt.xlabel('Year')
plt.ylabel('Percent')
plt.ylim(0, 1)
plt.show()

filtered_df = df_not_on_spotify[(df_not_on_spotify['score'] >= 8) & (df_not_on_spotify['new_album'] == 1) & (df_not_on_spotify['num_prev_reviews'] < 3)]



df_not_on_spotify[df_not_on_spotify['num_prev_reviews'] < 3].head()

# x: year
# y: score 8.0 - 10.0
# z: count of first/second releases with that score

filtered_df = df[(df['score'] >= 8) & (df['new_album'] == 1) & (df['num_prev_reviews'] == 1)]
temp_df = pd.DataFrame(filtered_df.groupby(['pub_year', 'score'])['reviewid'].count()).reset_index()
heatmap_df = temp_df.pivot(index='score', columns='pub_year', values='reviewid')

plt.figure(figsize=(16,8))
plt.title("First Time Pitchfork has Reviewed Artist")
plt.suptitle('Count of Albums Receiving Scores +8.0')
plt.xlabel('Year')
plt.ylabel('Score')
sns.heatmap(heatmap_df[::-1], 
            annot=(heatmap_df[::-1]).round(3), 
            norm=LogNorm(vmin=0, vmax=1), 
            fmt='g', 
            cmap=sns.light_palette((250, 80, 40), input='husl', n_colors=16)
           )
plt.show()

# x: year
# y: score 8.0 - 10.0
# z: count of first/second releases with that score

filtered_df = df_merged[(df_merged['score'] >= 8) & (df_merged['new_album'] == 1) & (df_merged['order_of_release'] < 3)]
temp_df = pd.DataFrame(filtered_df.groupby(['pub_year', 'score'])['best_new_music'].count()).reset_index()
heatmap_df = temp_df.pivot(index='score', columns='pub_year', values='best_new_music')

plt.figure(figsize=(16,8))
plt.title("First or Second Major Release by Artist")
plt.suptitle('Count of Albums Receiving Scores +8.0')
plt.xlabel('Year')
plt.ylabel('Score')
sns.heatmap(heatmap_df[::-1], 
            annot=(heatmap_df[::-1]).round(3), 
#             norm=LogNorm(vmin=0, vmax=1), 
            fmt='g', 
            cmap=sns.light_palette((250, 80, 40), input='husl', n_colors=16)
           )
plt.show()

df_merged

df_new_artist_bnm.groupby('pub_year')['title'].count()

df_merged[['artist', 'title', 'score', 'num_prev_reviews', 'order_of_release']]



