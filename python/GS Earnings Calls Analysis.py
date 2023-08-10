import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from datetime import datetime


get_ipython().magic('matplotlib inline')

filename = 'earningscalls/earningscalls/spiders/jpm_transcript_output.csv'

with open(filename) as f:
    content = f.readlines()

import ast
l = []
for item in content:
    try:
        l.append(ast.literal_eval(item))
    except:
        continue

for n, i in enumerate(l):
    if type(i) != dict:
        l[n] = dict(i[0])

import re

for i in l:
    for k, v in i.items():
        m = re.search('(Q[1-4])\s([0-9]*)', k)
        if m:
            found = m.group(2) + m.group(1)
            i[found] = i.pop(k)

quarters = []
for i in l:
    for k, v in i.items():
        quarters.append(k)

def remove_header_footer(lst):
    for n, i in enumerate(lst):
        if '><strong>Operator' in i:
            break
    lst = lst[n:]
    for n, i in enumerate(lst):
        if '<strong>Copyright policy' in i:
            break
    lst = lst[:n]
    return lst
                

# for n, i in enumerate(l[0]['2013Q4']):
#     print i

# for n, i in enumerate(l):
#     print n, i
#     break

import lxml.html
from collections import defaultdict
from fuzzywuzzy import fuzz

def create_dict(lst):
    d = defaultdict(list)
    k_list = []
    for n, i in enumerate(lst):
        if '><strong>' in i:
            try:
                key_html = lxml.html.fromstring(i)
                value_html = lxml.html.fromstring(lst[n+1])
            except:
                continue
            
            key = key_html.text_content()
            try:
                key = key.split()[0] + key.split()[1]
            except:
                key = key.split()[0]
            if key not in k_list:
                k_list.append(key)
            # fuzzy matching to resolve typos for keys
            for n, existing_key in enumerate(k_list):
                ratio = fuzz.ratio(existing_key, key)
                if ratio < 100 and ratio > 69:
                    key = existing_key
                    k_list.pop(n)
            value = value_html.text_content()
            #remove unicode characters 
            value = value.decode('unicode_escape').encode('ascii','ignore')

            d[key].append(value)
    return d
    

new_EC_dict = {}

for d in l:
    new_EC_dict[d.keys()[0]] = d.values()

# new_EC_dict['2014Q4']

df_EC = pd.DataFrame(new_EC_dict)

for k, v in new_EC_dict.items():
    new_EC_dict[k] = remove_header_footer(v[0])

def split_discussion_q_and_a(lst):
    discussion_l = []
    q_and_a_l = []
    for n, i in enumerate(lst):
        if 'id="question-answer-session"' in i:
            break
        discussion_l.append(i)
        q_and_a_l = lst[n:] 
    return discussion_l, q_and_a_l

q_and_a_dict = {}
discussion_dict = {}
for k, v in new_EC_dict.items():
    discussion_dict[k] = split_discussion_q_and_a(v)[0]
    q_and_a_dict[k] = split_discussion_q_and_a(v)[1]

for k, v in discussion_dict.items():
    lines = []
    for line in v:
        line = lxml.html.fromstring(line).text_content()
        line = line.decode('unicode_escape').encode('ascii','ignore')
        lines.append(line)
    discussion_dict[k] = lines

for k, v in q_and_a_dict.items():
    lines = []
    for line in v:
        line = lxml.html.fromstring(line).text_content()
        line = line.decode('unicode_escape').encode('ascii','ignore')
        lines.append(line)
    q_and_a_dict[k] = lines

q_and_a_dict['2013Q3']

dict_l = []
quarter_l = []
dates = []
for n, i in enumerate(l):
    try:
        key, value = i.items()[0]
        quarter_l.append(key)
        date = value[2]
        dates.append(date)
        lst = remove_header_footer(l[n][key])
        d = create_dict(lst)
        dict_l.append((key, d))
    except:
        print(n, key)
        continue

df_EC_speakers = pd.DataFrame(dict_l)
df_EC_speakers

from dateutil.parser import *

def clean_dates(dates):
    for n, date in enumerate(dates):
        date = lxml.html.fromstring(date).text_content().split()
        datetime_object = parse(' '.join(date[:3]))
        dates[n] = datetime_object
        

clean_dates(dates)
# zip(dates, quarters)

len(dict_l)

# remove Operator lines
def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

for n, d in enumerate(dict_l):
    d = removekey(d, 'Operator')
    dict_l[n] = d

dict_l[0]

# for n, d in enumerate(dict_l):
#     try:
#         print(d)
#     except:
#         continue

# speakers = []
# for n, d in enumerate(dict_l):
#     for k, v in d.items():
#         speakers.append(k)
# set(speakers)

# get eps surprises

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

import os

chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

PROXY = 'http://us-ny.proxymesh.com:31280'

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--proxy-server=%s' % PROXY)

driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)
# chrome.get("http://whatismyipaddress.com")
driver.get("https://www.streetinsider.com/ec_earnings.php?q=c")

# driver.implicitly_wait(5) # seconds


cookies = driver.get_cookies()

button=driver.find_element_by_xpath('//*[@id="expandbtn"]')
button.click()

surprise_quarters = []
consensus = []
surprise = []
for j in range(25):
    try:
        q = driver.find_element_by_xpath('//*[@id="content"]/table[1]/tbody/tr['+ str(j) + ']/td[3]')
        c = driver.find_element_by_xpath('//*[@id="content"]/table[1]/tbody/tr['+ str(j) + ']/td[5]')
        s = driver.find_element_by_xpath('//*[@id="content"]/table[1]/tbody/tr['+ str(j) + ']/td[6]') 
        surprise_quarters.append(q.text)
        consensus.append(c.text)
        surprise.append(s.text)
    except:
        continue
        
for j in range(25):
    try:
        q = driver.find_element_by_xpath('//*[@id="history_extra"]/tbody/tr[' + str(j) + ']/td[3]')
        c = driver.find_element_by_xpath('//*[@id="history_extra"]/tbody/tr[' + str(j) + ']/td[5]')
        s = driver.find_element_by_xpath('//*[@id="history_extra"]/tbody/tr[' + str(j) + ']/td[6]')
        
        surprise_quarters.append(q.text)
        consensus.append(c.text)
        surprise.append(s.text)
    except:
        continue

df_surprise = pd.DataFrame()
df_surprise['Quarters'] = pd.Series(surprise_quarters)
df_surprise['Surprise_Consensus_EPS'] = pd.Series(consensus)
df_surprise['Surprise_EPS'] = pd.Series(surprise)

# fix quarter format

def fix_eps_quarters(s):
    q = s[:2]
    year = s[2:]
    return '20' + year + q

# fix EPS to numerice

def fix_eps_to_numeric(s):
    s = s.replace("$", "")
    return s

df_surprise['Quarters'] = df_surprise.Quarters.apply(fix_eps_quarters)

df_surprise['Surprise_Consensus_EPS'] = pd.to_numeric(df_surprise.Surprise_Consensus_EPS.apply(fix_eps_to_numeric))
df_surprise['Surprise_EPS'] =  pd.to_numeric(df_surprise.Surprise_EPS.apply(fix_eps_to_numeric))

df_surprise.sort_values(by='Quarters').head()

transcripts[0][0]

import nltk
from textblob import TextBlob

transcripts = dict_l
# transcripts[0]
for n, i in enumerate(dict_l):
#     print(len(dict_l[n]))
    for k in dict_l[n]:
        print(k)
    break

sentiments = []
quarters = []
for n, i in enumerate(dict_l):
    doc_list = [item for k[1] in dict_l[n] for item in i]
    quarters.append(dict_l[n][0])

    EC_text = ' '.join(str(e) for e in doc_list)
    EC = TextBlob(EC_text)

#     sentiment = [i.sentiment.polarity for i in EC.sentences]
    sentiment = EC.sentiment.polarity

#     total_sentiment = 0.0
#     for i in sentiment:
#         total_sentiment = total_sentiment + i
    sentiments.append(sentiment)
pairs = zip(quarters, dates, sentiments)
pairs.sort()
pairs

unzipped_pairs = zip(*pairs)

df_sentiment = pd.DataFrame()
df_sentiment['Quarters'] = pd.Series(unzipped_pairs[0])
df_sentiment['EC_Dates'] = pd.Series(unzipped_pairs[1])
df_sentiment['EC_sentiment'] = pd.Series(unzipped_pairs[2])

df_surprise

df_merged = pd.merge(df_sentiment, df_surprise, how='left', on='Quarters')

import csv
filename = 'Cdailyprice2010_2016.csv'

Dates = []
Close = []
with open(filename) as f:
    content = csv.reader(f)
    for row in content:
        Dates.append(row[0])
        Close.append(row[4])
        

Dates = Dates[1:]
Close = Close[1:]
df = pd.read_csv('Cdailyprice2010_2016.csv')

df_merged.head()

# df2 = pd.DataFrame([['2013Q3', '2013-10-17', 0, 0, 0], ['2014Q2', '2014-07-15', 0, 0, 0]], columns=['Quarters', \
#                             'EC_Dates', 'EC_sentiment', 'Surprise_Consensus_EPS', 'Surprise_EPS'])

# df_merged = df_merged.append(df2, ignore_index=True)
df_merged["EC_Dates"] = pd.to_datetime(df_merged.EC_Dates, errors='coerce')
df_merged = df_merged.sort_values(by='Quarters')

# df_merged.Surprise_EPS.plot(kind='bar')

objects = df_merged.Quarters

quarters = np.arange(df_merged.Quarters.shape[0])
surprises = df_merged.Surprise_EPS

plt.figure(figsize=(10,5))
plt.bar(quarters, surprises, alpha=0.5, align='center')
plt.xticks(quarters, objects, rotation='vertical')
plt.ylabel('EPS Surprise ($)')
plt.title('C Earnings Surprise')
plt.autoscale()
plt.show()

objects = df_merged.Quarters

quarters = np.arange(df_merged.Quarters.shape[0])
sentiments = df_merged.EC_sentiment

plt.figure(figsize=(10,5))
plt.bar(quarters, sentiments, alpha=0.5, align='center')
plt.xticks(quarters, objects, rotation='vertical')
plt.ylabel('Sentiment')
plt.title('C Earnings Call Sentiment Analysis')
plt.autoscale()
plt.show()

df = df.sort_values(['Date'])

df.Date = pd.to_datetime(df['Date'])
EC_dates_mask = df_merged['EC_Dates']

EC_dates_mask_index = []
for n, i in enumerate(df.Date):
    if i in list(EC_dates_mask):
        EC_dates_mask_index.append(n)

markers_on = EC_dates_mask_index
plt.figure(figsize=(10,5))
plt.ylabel('Share Price ($)')
plt.title('C Share Price')
plt.plot(df.Date, df.Close, '-')
plt.plot(df.Date, df.Close, 'rD', markevery=markers_on)


# plt.autoscale()

df_scatter = pd.DataFrame()
df_scatter = pd.merge(df_merged, df, left_on='EC_Dates', right_on='Date')
df_scatter['Day_percent_change'] = (df_scatter.Close - df_scatter.Open)/df_scatter.Open * 100
# df_scatter = df_scatter.drop(12)
# df_scatter = df_scatter.drop(15)
df_scatter.head()

plt.figure(figsize=(10,5))
plt.ylabel('End of Day Closing Price ($)')
plt.xlabel('Earnings Call Sentiment')
plt.title('C End of Day Closing Price vs. Earnings Call Sentiment')
plt.scatter(df_scatter.EC_sentiment, df_scatter.Close)
# plt.scatter(df_scatter.EC_sentiment, df_scatter.Surprise_EPS)

# for word, count in sorted(EC.word_counts.items(),key=lambda x: x[1], reverse=True):
#     print("%15s %i" % (word,count))

from nltk.util import ngrams

from collections import defaultdict
from operator import itemgetter

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop += ['.', ',', '(', ')', "'", '"']

counter = defaultdict(int)

n = 2
# for doc in documents:
words = TextBlob(EC_text).words
words = [w for w in words if w not in stop]
bigrams = ngrams(words,n)
for bigram in bigrams:
    counter[bigram] += 1
            
for bigram, count in sorted(counter.items(), key = itemgetter(1), reverse=True)[:30]:
    phrase = " ".join(bigram)
#     print('%20s %i' % (phrase, count))

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from __future__ import print_function

t_dict_corpus = zip(*dict_l)[1]
l_dict_corpus = list(t_dict_corpus)

# discussion_corpus = []
# for k,v in discussion_dict.items():
#     for item in v:
#         discussion_corpus.append((k, item))
        
# q_and_a_corpus = []
# for k,v in q_and_a_dict.items():
#     for item in v:
#         q_and_a_corpus.append((k, item))
    
    

# discussion_corpus = zip(*discussion_corpus)[1]
# q_and_a_corpus = zip(*q_and_a_corpus)[1]

# discussion_corpus[2]
# q_and_a_corpus[4]

corpus = []
for n, i in enumerate(l_dict_corpus):
    doc_list = [item for k,v in l_dict_corpus[n].items() for item in v]
    EC_text = ' '.join(str(e) for e in doc_list)
    corpus.append(EC_text)

# len(corpus)

# corpus

n_samples = 2000
n_features = 1000
n_topics = 7
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# my_stops

my_stops = stopwords.words('english')
my_stops = my_stops + ['ahead', 'youre', 'weve', 'yeah', 'hi', 'hey', 'im', 'youve', 'theres', 'indiscernible',                      'thats', 'theyre', 'youll', 'david', 'roger', 'freeman', 'please', 'harvey', 'schwartz',                      'fiona', 'swaffield', 'operator', 'glenn', 'officer', 'executive', 'vice', 'president',                      'mitchell', 'michael', 'mayo', 'securities', 'agricole', 'morning', 'mike', 'steven',                      'christian', 'devin', 'guy', 'marty', 'betsy', 'jim', 'chubak', 'nomura', 'graseck',                      'morgan', 'stanley', 'kian', 'ph', 'bruce']

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words=my_stops, max_features=n_features, ngram_range=(1,2))
cv_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=my_stops, max_features=n_features)
tfidf_vectorizer 
cv_vectorizer

tfidf = tfidf_vectorizer.fit_transform(corpus)
cv = cv_vectorizer.fit_transform(corpus)

nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5)

topic_vectors = nmf.fit_transform(cv)

print("\nTopics in NMF model:")
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
cv_feature_names = cv_vectorizer.get_feature_names()
# print_top_words(nmf, tfidf_feature_names, n_top_words)
print_top_words(nmf, cv_feature_names, n_top_words)

np.set_printoptions(precision=2)

sort_topicvectors_quarters = []

for i in zip(list(df_EC_speakers[0]), topic_vectors):
    sort_topicvectors_quarters.append(i)

# for i in sorted(sort_topicvectors_quarters):
#     print(i)

# df_nmf = pd.DataFrame(topic_vectors)
# df_nmf

sort_topicvectors_quarters.sort()
quarters, topics = zip(*sort_topicvectors_quarters)

max_topics = []
for i in topics:
    max_topic = np.argmax(i)
    max_topics.append(max_topic)
    
plt.figure(figsize=(10,5))
plt.bar(range(len(quarters)), max_topics, align='center')
plt.xticks(range(len(quarters)), quarters, rotation='vertical')
plt.ylabel('Topic Category')
plt.xlabel('Quarters')
plt.title('Topic Modelling by Quarter')
plt.autoscale()
plt.show()

topic_vectors.shape

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tfidf)

print("\nTopics in LDA model:")
tf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

topic_vectors = lda.fit_transform(tfidf)

np.set_printoptions(precision=2)

sort_topicvectors_quarters = []

for i in zip(list(df_EC_speakers[0]), topic_vectors):
    sort_topicvectors_quarters.append(i)

for i in sorted(sort_topicvectors_quarters):
    print(i)



