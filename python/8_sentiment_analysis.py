import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

get_ipython().magic('matplotlib inline')

pd.options.display.max_rows = None
pd.options.display.max_colwidth = -1

with open('lda_top_tweets.pkl', 'rb') as pklfile:
    top_tweets = pickle.load(pklfile)

top_tweets['polarity'] = top_tweets.text.apply(lambda x: TextBlob(x).sentiment.polarity)
top_tweets['subjectivity'] = top_tweets.text.apply(lambda x: TextBlob(x).sentiment.subjectivity)
top_tweets['pol_z'] = top_tweets.groupby(['community','topic']).polarity.transform(lambda x: (x-np.mean(x))/np.std(x))

# Examine various topics, their tweet text, and their polarity scores
top_tweets.loc[(top_tweets.community == '10') & (top_tweets.topic == 0),                ['screen_name', 'name', 'text', 'subjectivity', 'polarity', 'pol_z']].     sort_values('pol_z')

# Stringify my topic ids
top_tweets['topic_num'] = top_tweets.topic
top_tweets.topic = top_tweets.topic.apply(lambda x: str(x))

# Name topics that are coherent, NaN topics that are muddy
top_tweets.loc[top_tweets.community == '10', 'topic'] = top_tweets.loc[top_tweets.community == '10', 'topic'].     replace(['0','1','2','3','4','5','6','7','8','9'], ['shell','','tillerson','reef','permafrost','',
                                                        'melt','budget','group17','pruitt'])
    
top_tweets.loc[top_tweets.community == '3', 'topic'] = top_tweets.loc[top_tweets.community == '3', 'topic'].     replace(['0','1','2','3','4','5','6','7'], ['nye','pruitt','','budget','','','',''])
    
top_tweets.loc[top_tweets.community == '4', 'topic'] = top_tweets.loc[top_tweets.community == '4', 'topic'].     replace(['0','1','2','3','4','5','6','7'], ['pruitt','women','energy','','','','reef',''])
    
top_tweets.loc[top_tweets.community == '2', 'topic'] = top_tweets.loc[top_tweets.community == '2', 'topic'].     replace(['0','1','2','3','4','5','6'], ['','pruitt','','','nye','budget',''])

top_tweets.loc[top_tweets.community == '18', 'topic'] = top_tweets.loc[top_tweets.community == '18', 'topic'].     replace(['0','1','2','3','4','5','6','7'], ['','budget','','tillerson','reef','','pruitt',''])

top_tweets.loc[top_tweets.community == '12', 'topic'] = top_tweets.loc[top_tweets.community == '12', 'topic'].     replace(['0','1','2','3','4','5','6','7'], ['','reef','inuit','budget','','','pruitt',''])

top_tweets.loc[top_tweets.community == '28', 'topic'] = top_tweets.loc[top_tweets.community == '28', 'topic'].     replace(['0','1','2','3','4','5','6'], ['shell','uk','budget','reef','pruitt','',''])

top_tweets.loc[top_tweets.community == '9', 'topic'] = top_tweets.loc[top_tweets.community == '9', 'topic'].     replace(['0','1','2','3','4','5','6','7'], ['','pruitt','budget','','emissions','','','tillerson'])

# Which topics does each community talk about?
top_tweets.groupby(['community', 'topic']).size()

# How does average sentiment differ between communities and topics?
top_tweets.groupby(['community', 'topic']).polarity.agg(['mean', 'std']).reset_index().     sort_values(['topic','community'], ascending=False)

# Create charts
fig = plt.figure(figsize=(8,6))
top_tweets[(top_tweets.topic=='pruitt') & (top_tweets.subjectivity != 0)].groupby('community').     polarity.mean().plot(kind='bar')
plt.ylabel('Avg. Positivity Score', fontsize=15)
plt.xlabel('Community', fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(np.arange(8),('Global Mix', 'Canada','Acad./Authors','Journalists','UK','Deniers',
                         'CC Orgs','Celebs'), rotation=-45, fontsize=14)
plt.title('Scott Pruit Denies That CO2 Causes Global Warming', fontsize=16).     set_position([.5, 1.05]);

fig = plt.figure(figsize=(8,6))
top_tweets[(top_tweets.topic=='nye') & (top_tweets.subjectivity != 0)].groupby('community').     polarity.mean().plot(kind='bar')
plt.ylabel('Avg. Positivity Score', fontsize=15)
plt.xlabel('Community', fontsize=15)
plt.xticks(np.arange(2), ('Journalists', 'Deniers'), rotation=0, fontsize=14)
plt.yticks(fontsize=12)
plt.ylim(-.15, .16)
plt.title('Bill Nye / Tucker Carlson Debate', fontsize=16).set_position([.5,1.05])

top_tweets[(top_tweets.topic=='budget') & (top_tweets.subjectivity != 0)].groupby('community').     polarity.mean().plot(kind='bar')

top_tweets.loc[(top_tweets.community == '3') & (top_tweets.topic == 'nye'),                ['screen_name', 'name', 'text', 'subjectivity', 'polarity', 'pol_z']].     sort_values('pol_z')
    
# 'MrLukeyLuke' is in the Denier community, but is much more positive on Bill Nye than the average.
# He appears to accept climate change science.

top_tweets.loc[(top_tweets.community == '3') & (top_tweets.topic == 'pruitt'),                ['screen_name', 'name', 'text', 'subjectivity', 'polarity', 'pol_z']].     sort_values('pol_z')
    
# devoncatt is in the Denier community, but is much more negative than average about Scott Pruitt's CO2 comments
# She appears to accept climate change science



