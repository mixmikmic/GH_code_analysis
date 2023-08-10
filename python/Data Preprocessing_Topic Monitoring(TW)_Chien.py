import pandas as pd
import numpy as np
import re
import nltk
import csv
import sys
# from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import time
import os
from IPython.display import Image
from IPython.display import display
# os.remove('final_twitter_preprocessing.csv')
# os.remove('twitter_preprocessing.csv')

### Load the Twitter Dataset
# Remove duplicates, NA, preserve only English texts
disease = pd.read_csv('TW_Tweet.csv', encoding = 'UTF-8', low_memory = False)
df = pd.DataFrame(disease, columns = ['id', 'keyword', 'created', 'language', 'message'])
df.columns = ['id', 'key', 'created_time', 'language', 'message']
rm_duplicates = df.drop_duplicates(subset = ['key', 'message'])
rm_na = rm_duplicates.dropna()
dtime = rm_na.sort_values(['created_time'])
dtime.index = range(len(dtime))
dlang = dtime[dtime['language'] == 'en']
dlang = dlang[dlang['key'] != 'johnson & johnson']
dlang = dlang[dlang['key'] != 'johnson&johnson']
dlang.index = range(len(dlang))
display(dlang.head(3))
print(len(dlang))

### First need to login and get the Yandex API key from https://tech.yandex.com/translate/

import json
import requests
from urllib.request import urlopen

# Add your own key here
api_key = "************"

# Detect the language of text
def get_translation_direction(api_key,text):
    url = "https://translate.yandex.net/api/v1.5/tr.json/detect?"
    url = url + "key=" + api_key
    if(text != ""):
        url = url+"&text="+text
    r = requests.get(url)
    return (r.json()['lang'])
    
# Translate the text into English
def translation(api_key,text,lang):
    url = "https://translate.yandex.net/api/v1.5/tr.json/translate?"
    url = url + "key=" + api_key
    if(text != ""):
        url = url + "&text=" + text
    if(lang != ""):
        url = url + "&lang=" + lang
    r = requests.get(url)
    print(''.join(r.json()['text']))
    return(''.join(r.json()['text']))
    
# Add the text you want to detect and the language you want to translate
# For lang, you can check here to see the code of language you want to translate https://tech.yandex.com/translate/doc/dg/concepts/api-overview-docpage/
# Below is an example for language translation process
text = "Do you know that she is coming?"
lang = "de"
print("Language Detection:")
print(get_translation_direction(api_key, text), ',', text)
print("Translation:")
print(lang, ',', translation(api_key, text, lang))

import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import string
import time

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Create a new csv file to store the result after data preprocessing
with open(
        'twitter_preprocessing.csv',
        'w',
        encoding = 'UTF-8',
        newline = '') as csvfile:
    column = [[
        'id', 'key', 'created_time', 'language', 'message', 're_message'
    ]]
    writer = csv.writer(csvfile)
    writer.writerows(column)
    
# Data preprocessing steps   
for i in range(len(dlang['message'])):
    features = []
    features.append(str(int(dlang['id'][i])))
    features.append(dlang['key'][i])
    features.append(dlang['created_time'][i])
    features.append(dlang['language'][i])
    features.append(dlang['message'][i])
    reurl = re.sub(r"http\S+", "", str(dlang['message'][i]))
    tokens = ' '.join(re.findall(r"[\w']+", reurl)).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    x = ' '.join(x)
    stop_free = " ".join([i for i in x.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word,pos = 'n') for word in punc_free.split())
    normalized = " ".join(lemma.lemmatize(word,pos = 'v') for word in normalized.split())
    word = " ".join(word for word in normalized.split() if len(word)>3)
    postag = nltk.pos_tag(word.split())
    irlist = [',','.',':','#',';','CD','WRB','RB','PRP','...',')','(','-','``','@']
    poslist = ['NN','NNP','NNS','RB','RBR','RBS','JJ','JJR','JJS']
    wordlist = ['co', 'https', 'http','rt','com','amp','fe0f','www','ve','dont',"i'm","it's",'isnt','âźă','âąă','âł_','kf4pdwe64k']
    adjandn = [word for word,pos in postag if pos in poslist and word not in wordlist and len(word)>3]
    stop = set(stopwords.words('english'))
    wordlist = [i for i in adjandn if i not in stop]
    features.append(' '.join(wordlist))
    with open('twitter_preprocessing.csv', 'a', encoding = 'UTF-8', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])
df_postncomment = pd.read_csv('twitter_preprocessing.csv', encoding = 'UTF-8', sep = ',')
df_rm = df_postncomment.drop_duplicates(subset=['id', 're_message'])
rm_english_na = df_rm.dropna()
rm_english_na.index = range(len(rm_english_na))
dfinal_tw = pd.DataFrame(
    rm_english_na,
    columns = ['id', 'key', 'created_time', 'language', 'message', 're_message'])
dfinal_tw.to_csv(
    'final_twitter_preprocessing.csv',
    encoding = 'UTF-8',
    columns = ['id', 'key', 'created_time', 'language', 'message', 're_message'])
os.remove('twitter_preprocessing.csv')

test = pd.read_csv('final_twitter_preprocessing.csv', encoding = 'UTF-8', sep = ',', index_col = 0)
display(test.head(3))
print(len(test))

