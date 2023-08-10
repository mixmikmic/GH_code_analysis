import requests
import numpy as np
import pandas as pd
import json
from IPython.display import display
import re
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from lib.twitter_keys import keys

sql_select = '''
select * from tweets_demo where lang = 'en'
'''

import psycopg2 as pg2
import psycopg2.extras as pgex
this_host='54.69.228.16'
this_user='postgres'
this_password='postgres'
conn = pg2.connect(host = this_host, 
                        user = this_user,
                        password = this_password)
cur = conn.cursor(cursor_factory=pgex.RealDictCursor)
#cur.execute(sql_create)
#cur.execute(sql_drop)
#cur.execute(sql_insert)
#conn.commit()
cur.execute(sql_select)
rows = cur.fetchall()
conn.close()
df = pd.DataFrame(rows)

len(df)

list(df.columns.values)

import pickle
get_ipython().system('pip install redis')
import redis
redis_ip = '52.37.247.211'
r = redis.StrictRedis(redis_ip)
#r.flushall()

def c(text):
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    return text

from spacy.en import English
nlp = English()

content_vecs = np.array([nlp(i).vector for i in df['tweet_content']])
content_vecs.shape

type(content_vecs[0])

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=6500)
nn.fit(content_vecs)

def most_similar(search):
    distance, indices = nn.kneighbors(nlp(search).vector.reshape(1,-1))
    
    return df.ix[indices[0]][['location','created_at','tweet_content','trump']]

cat_sport = most_similar('sports')
cat_sport

cat_music = most_similar('music')
cat_music

cat_movies = most_similar('movies')
cat_movies

cat_travel = most_similar('travel')
cat_travel

cat_TV = most_similar('TV')
cat_TV 

cat_tech = most_similar('technology')
cat_tech

cat_health = most_similar('health')
cat_health 

cat_food = most_similar('food')
cat_food

cat_business = most_similar('busniess')
cat_business

cat_gov = most_similar('government')
cat_gov 

cat_law = most_similar('law')
cat_law

cat_education= most_similar('education')
cat_education

cat_entertainment = most_similar('entertainment')
cat_entertainment

cat_fashion = most_similar('fashion')
cat_fashion

cat_shopping = most_similar('shopping')
cat_shopping

cat_song = most_similar('song')
cat_song

#nothing related come up# 
cat_trump = most_similar('trump')
cat_trump

sum(cat_gov.index.isin(cat_law.index) == True)/6500

sum(cat_music.index.isin(cat_movies.index) == True)/6500

sum(cat_music.index.isin(cat_song.index) == True)/6500

sum(cat_entertainment.index.isin(cat_song.index) == True)/6500

sum(cat_TV.index.isin(cat_movies.index) == True)/6500



sum(cat_gov.index.isin(cat_business.index) == True)/6500



cat_music.index.isin(cat_entertainment.index) == True

sum(cat_music.index.isin(cat_entertainment.index) == True)/6500

sum(cat_movies.index.isin(cat_entertainment.index) == True)/5000

sum(cat_fashion.index.isin(cat_entertainment.index) == True)/6500

sum(cat_fashion.index.isin(cat_shopping.index) == True)/6500

sum(cat_business.index.isin(cat_shopping.index) == True)/6500

sum(cat_business.index.isin(cat_education.index) == True)/6500

sum(cat_education.index.isin(cat_gov.index) == True)/6500

sum(cat_trump.index.isin(cat_entertainment.index) == True)/6500

sum(cat_trump.index.isin(cat_gov.index) == True)/6500

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

import redis
redis_ip = '34.210.97.79'
content_vecs = pickle.loads(r.get('nlp_content_vecs'))
cat_vecs = pickle.loads(r.get('nlp_cat_vecs'))
def Similarity_score(text):
    wiki_cat_df = pd.DataFrame(list(cli.wiki_mongo_database.wiki_cat_collection.find({})))
    #cat_vecs = np.array([nlp(i).vector for i in wiki_cat_df['text']])
    r = redis.StrictRedis(redis_ip)
    cat_vecs = pickle.loads(r.get('nlp_cat_vecs'))
    similarity_score={}
    for j in range(len(cat_vecs)):
        similarity = cosine_similarity(cat_vecs[j].reshape(1,-1),nlp(text).vector.reshape(1,-1))
        similarity_score[(wiki_cat_df['category'][j])] = round(similarity[0][0],3)
        cs_s_df = pd.DataFrame.from_dict(similarity_score,orient='index')
        cs_s_df.columns = ['score']
        cs_s_df = cs_s_df.sort_values('score',ascending=False)
    return cs_s_df.head(3)

#for j in range(len(content_vecs)):
for j in range(20):
    similarity = cosine_similarity(content_vecs[j].reshape(1,-1),nlp('music').vector.reshape(1,-1))
    print(df['tweet_content'].iloc[j], round(similarity[0][0],3))

def c(text):
    text = re.sub("'","''", text)
    text = re.sub("{","\{",text)
    text = re.sub("}","\}",text)
    return text

from tqdm import tqdm
import psycopg2 as pg2
import psycopg2.extras as pgex
this_host='54.69.228.16'
this_user='postgres'
this_password='postgres'
conn = pg2.connect(host = this_host, 
                        user = this_user,
                        password = this_password)

for i in tqdm(range(len(df))):
    tweet = c(df['tweet_content'].iloc[i])
    similarity = cosine_similarity(content_vecs[i].reshape(1,-1),nlp('business').vector.reshape(1,-1))
    cur = conn.cursor()
    sql_update = '''
    update tweets_demo
    set business = '{}'
    where tweet_content = '{}';
    commit;
    '''.format(round(similarity[0][0],2),tweet)
    cur.execute(sql_update)
    conn.commit() 
    print(round(similarity[0][0],2), df['tweet_content'].iloc[i])
conn.close  

df[['screen_name','tweet_content','cleaned_tweet','music','location','trump','immigrant','climate']].    sort(columns='music', axis=0,ascending=False)

df[['screen_name','tweet_content','cleaned_tweet','music','location','trump','immigrant','climate']][df['music'] > 0.50].sort_values(by='music', axis=0,ascending=False)

'business','govt','movies','sports','tech','tv',

