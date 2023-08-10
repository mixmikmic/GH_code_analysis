import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy
import re
color = sns.color_palette()
get_ipython().magic('matplotlib inline')

new_df = pd.read_json("/Users/jakebialer/Neuromancers-Kaggle/price_vs_median30.json")

t_df = pd.read_json("/Users/jakebialer/Neuromancers-Kaggle/train.json")
t_df['description'].value_counts()

t_df.to_json("price_vs_median",force_ascii=False)
print new_df.shape,t_df.shape

get_ipython().magic('matplotlib inline')

sns.boxplot(x='interest_level', y='price_vs_median_30', data=new_df)

new_df.shape

t_df['has_blank_description']= t_df['description'].apply(lambda x: x.strip()) ==""
t_df['has_blank_description'].value_counts()
sns.countplot(x='interest_level',  hue='has_blank_description', data=t_df)

t_df.loc[10,'description']
t_df.loc[100004,'description']

no_dups_description=t_df[['description']].drop_duplicates()
no_dups_description

import string
from nltk import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    return tokens

def find_ngrams(df):
    word_vectorizer = CountVectorizer(ngram_range=(2,4), analyzer='word', stop_words='english',tokenizer=tokenize, max_features=200)
    sparse_matrix = word_vectorizer.fit_transform(df['description'])
    frequencies = sum(sparse_matrix).toarray()[0]
    return pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency']).sort('frequency',ascending=[0])


find_ngrams(no_dups_description)

phone_regex = "(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})" # http://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
has_phone = t_df['description'].str.extract(phone_regex)
t_df['has_phone']=[type(item)==unicode for item in has_phone]

t_df['has_phone'].value_counts()

sns.countplot(x='interest_level',  hue='has_phone', data=t_df)
has_phone_interest_level= t_df.groupby(['has_phone','interest_level'])['interest_level'].count()
phone_interest_level_pcts = has_phone_interest_level.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
phone_interest_level_pcts
# If have phone, looks like more likely to have higher interest


t_df['listing_length']=t_df['description'].str.len()

sns.violinplot(x='interest_level',  y='listing_length', data=t_df)

import re 
# http://stackoverflow.com/questions/520031/whats-the-cleanest-way-to-extract-urls-from-a-string-using-python
URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|edu|org|gov|ly)\b/?(?!@)))"""

def get_urls(text):
    urls = re.findall(URL_REGEX, text)
    return urls

    
    
t_df['url'] = t_df['description'].apply(get_urls)

t_df['url_count'] =t_df['url'] .apply(len)

t_df['url_count'].value_counts()

sns.violinplot(x='interest_level',  y='url_count', data=t_df)

#TODO
# Duplicate
duplicates = t_df.groupby(['description']).size().reset_index().rename(columns={0:'duplicates'}).sort(['duplicates'], ascending=[0])

t_df=pd.merge(t_df, duplicates, left_on='description', right_on='description')






sns.violinplot(x='interest_level',  y='duplicates', data=t_df)

# Guarantors
has_guarantors= t_df['description'].str.contains( "guarantor" ,case=False)
t_df['has_guarantors'] = has_guarantors
sns.countplot(x='interest_level',  hue='has_guarantors', data=t_df)
has_guarantors_interest_level= t_df.groupby(['has_guarantors','interest_level'])['interest_level'].count()
guarantors_interest_level= has_guarantors_interest_level.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
guarantors_interest_level

# Amount of CAPS 

def count_caps(message):
    caps =sum(1 for c in message if c.isupper())
    total_characters =sum(1 for c in test if c.isalpha())
    if total_characters>0:
        caps = caps/(total_characters* 1.0)
    return caps

t_df['amount_of_caps']=t_df['description'].apply(count_caps)





# Subway Line 
# From brokerage 
# MX id 
# Listing Id 
# Email Address
# Social Media
# has stainless steel
# has hardwood floors
# has broker
# spell check analysis 
# has html tags
# has new
# has free month
# has high celings
# has closet space
# pets
# equal housing

# sentiment analyis
from collections import OrderedDict, defaultdict, Counter
from nltk.tokenize.treebank import TreebankWordTokenizer
import csv
# https://github.com/pmbaumgartner/text-feat-lib/blob/master/notebooks/NRC%20Emotion%20Lexicon%20Features.ipynb
wordList = defaultdict(list)
emotionList = defaultdict(list)
with open('/Users/jakebialer/Desktop/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if int(present) == 1:
            #print(word)
            wordList[word].append(emotion)
            emotionList[emotion].append(word)


twt = TreebankWordTokenizer()

def generate_emotion_count(string, tokenizer):
    emoCount = Counter()
    for token in twt.tokenize(string):
        token = token.lower()
        emoCount += Counter(wordList[token])
    return emoCount

descriptions = t_df['description'].tolist()
emotionCounts = [generate_emotion_count(description, twt) for description in descriptions]
emotion_df = pd.DataFrame(emotionCounts)
emotion_df = emotion_df.fillna(0)
emotion_df

from textblob import TextBlob

descriptions = t_df['description'].tolist()
descriptions[0]

# 
from geopy.distance import vincenty
lat = t_df['latitude'].tolist()
long_ = t_df['longitude'].tolist()
midtown_lat = 40.7586
midtown_long = -73.9838
distance =[]
for i in range(len(lat)):
    distance.append(vincenty((lat[i],long[i]),(midtown_lat,midtown_long)).meters)
t_df['distance'] = distance

sorted(distance,reverse=True)

t_df['distance'] = distance

sns.boxplot(x='interest_level',  y='distance', data=t_df)

t_df = t_df[['latitude','longitude','price','bedrooms','bathrooms']]
t_df.shape[0]

def nearest_neighbors(df,n):
    # Input: df and num of meighbors
    # Output: df with price_vs_median for each row
    df_sub = df[['latitude','longitude','price','bedrooms','bathrooms']]
    rows = range(df.shape[0])
    diffs = map(lambda row: compare_price_vs_median(df_sub,n,row),rows)
    df['price_vs_median_'+str(n)] = diffs
    return df 


def compare_price_vs_median(df,n,i):
    # Help Function For nearest_neighbors
    # for each lat long 
    # calculate dist from all other lat longs with same beds and bathrooms 
    # find n nearest neighbors 
    # calculate median price of n nearest neighbors 
    # compare price vs median 

    print(i)
    row = df.iloc[i,:]
    lat = row['latitude']
    lon = row['longitude']
    bed = row['bedrooms']
    bath = row['bathrooms']
    price = row['price']
    df.index = range(df.shape[0])
    all_other_data = df.drop(df.index[[i]])
    with_same_bed_bath=all_other_data[all_other_data['bedrooms']==bed]
    with_same_bed_bath=with_same_bed_bath[with_same_bed_bath['bathrooms']==bath]
    longs = with_same_bed_bath['longitude'].tolist()
    lats = with_same_bed_bath['latitude'].tolist()
    distances = []
    for j in range(len(lats)):
        distance = vincenty((lats[j],longs[j]),(lat,lon)).meters
        distances.append(distance)
    # http://stackoverflow.com/questions/13070461/get-index-of-the-top-n-values-of-a
    dist_positions= sorted(range(len(distances)), key=lambda k: distances[k])[-n:] 
    top_dist_df= with_same_bed_bath.iloc[dist_positions,:]  
    med_price = with_same_bed_bath['price'].median()
    diff = price/med_price
    return diff

nearest_neighbors(t_df,10)

df = t_df[['latitude','longitude','price','bedrooms','bathrooms']]

rows = t_df.shape[0]

i=range(rows)[0]

row = df.iloc[i,:]

row

lat = row['latitude']
lon = row['longitude']
bed = row['bedrooms']
bath = row['bathrooms']
price = row['bathrooms']

lat

all_other_data = df.iloc[-i,:]

with_same_bed_bath=all_other_data[all_other_data['bedrooms']==bed and all_other_data['bathrooms']==bath]

with_same_bed_bath

all_other_data

all_other_data = df.iloc[-i,:]

all_other_data = df.ix[df.iloc[i,:],]

all_other_data

df.index = range(df.shape[0])
df

all_other_data = df.drop(df.index[[i]])

all_other_data

with_same_bed_bath=all_other_data.loc[all_other_data['bedrooms']==bed]

all_other_data['bedrooms']==bed
all_other_data['bathrooms']==bath



type(all_other_data['bedrooms'])

with_same_bed_bath=all_other_data[all_other_data['bedrooms']==bed]

with_same_bed_bath=with_same_bed_bath[with_same_bed_bath['bathrooms']==bath]

longs = with_same_bed_bath['longitude'].tolist()

lats = with_same_bed_bath['latitude'].tolist()

distances = []

lats

distances=[]
for j in range(len(lats)):
    distance = vincenty((lats[j],longs[j]),(lat,lon)).meters
    distances.append(distance)
distances

n=25

dist_positions= sorted(range(len(distances)), key=lambda k: distances[k])[-n:] 

dist_positions

top_dist_df= with_same_bed_bath.iloc[dist_positions,:] 

top_dist_df

med_price = with_same_bed_bath['price'].median()

med_price



diff

price = row['price']

diff = price/med_price

diff

t_df['price'].describe()

count_nan = len(t_df['price']) - t_df['price'].count()

count_nan



