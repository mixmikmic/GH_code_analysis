import pandas as pd
import numpy as np
import re
import csv
from langdetect import detect
import nltk
# nltk.download('punkt')
# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from IPython.display import Image
from IPython.display import display

### Load the Crawled Facebook Dataset
# Remove duplicates, NA, sorted time
disease = pd.read_csv('Final_utf16.csv', encoding = 'utf-16LE', sep=',',
                         dtype={"key": object, "id.x": object,"like_count.x": float, "from_id.x":float,
                                "from_name.x":object, "message.x":object, "created_time.x":object, "type":object,
                                "link":object, "story":object, "comments_count.x":float,"shares_count":float,
                                "love_count":float, "haha_count":float, "wow_count":float, "sad_count": float,
                                "angry_count":float, "join_id":object, "from_id.y":float, "from_name.y":object,
                                "message.y":object, "created_time.y":object, "likes_count.y":float, 
                                "comments_count.y": float, "id.y":object})
df = pd.DataFrame(disease, columns=['key', 'created_time.x', 'id.x','message.x' , 'id.y', 'message.y'])
df.columns = ['key', 'created_time.x', 'id.x','message.x' , 'id.y', 'message.y']
rm_duplicates = df.drop_duplicates(subset=['message.x', 'message.y'])
dtime = rm_duplicates.sort_values(['created_time.x'])
dtime.index = range(len(dtime))
dlang = dtime
dlang = dlang[dlang['key']!='johnson & johnson']
dlang = dlang[dlang['key']!='johnson&johnson']
dlang.index = range(len(dlang))
display(dlang.head(3))
print(len(dlang))

# Detect the text language by majority vote
def calculate_languages_ratios(text):
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements)
    return languages_ratios
    
def detect_language(text):
    ratios = calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language

import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import string
import time
import os
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Create a new csv file to store the result after data preprocessing
with open('facebook_preprocessing.csv', 'w', encoding = 'UTF-8', newline = '') as csvfile:
    column = [['key', 'created_time.x', 'id.x', 'message.x', 'id.y', 'message.y',
               'lang.x', 're_message.x', 'lang.y', 're_message.y']]
    writer = csv.writer(csvfile)
    writer.writerows(column)

# Data preprocessing steps
for i in range(len(dlang['message.x'])): 
    features = []
    features.append(dlang['key'][i])
    features.append(dlang['created_time.x'][i])
    features.append(dlang['id.x'][i])
    features.append(dlang['message.x'][i])
    features.append(dlang['id.y'][i])
    features.append(dlang['message.y'][i])
    if(str(dlang['message.x'][i]) == "nan"):
        features.append('english')
        features.append(dlang['message.x'][i])
    else:
        lang = detect_language(dlang['message.x'][i])
        features.append(lang)
        stop = set(stopwords.words(lang))
        reurl = re.sub(r"http\S+", "", str(dlang['message.x'][i]))
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
        stop = set(stopwords.words(lang))
        wordlist = [i for i in adjandn if i not in stop]
        features.append(' '.join(wordlist))
    if(str(dlang['message.y'][i]) == "nan"):
        features.append('english')
        features.append(dlang['message.y'][i])
    else:
        lang = detect_language(dlang['message.y'][i])
        features.append(lang)
        stop = set(stopwords.words(lang))
        reurl = re.sub(r"http\S+", "", str(dlang['message.y'][i]))
        tokens = ' '.join(re.findall(r"[\w']+", reurl)).lower().split()
        x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
        x = ' '.join(x)
        stop_free = " ".join([i for i in x.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word,pos='n') for word in punc_free.split())
        normalized = " ".join(lemma.lemmatize(word,pos='v') for word in normalized.split())
        word = " ".join(word for word in normalized.split() if len(word)>3)
        postag = nltk.pos_tag(word.split())
        irlist = [',','.',':','#',';','CD','WRB','RB','PRP','...',')','(','-','``','@']
        poslist = ['NN','NNP','NNS','RB','RBR','RBS','JJ','JJR','JJS']
        wordlist = ['co', 'https', 'http','rt','com','amp','fe0f','www','ve','dont',"i'm","it's",'isnt','âźă','âąă','âł_','kf4pdwe64k']
        adjandn = [word for word,pos in postag if pos in poslist and word not in wordlist and len(word)>3]
        stop = set(stopwords.words(lang))
        wordlist = [i for i in adjandn if i not in stop]
        features.append(' '.join(wordlist))
    with open('facebook_preprocessing.csv', 'a', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])
        
df_postncomment = pd.read_csv('facebook_preprocessing.csv', encoding = 'UTF-8', sep = ',')
rm_na = df_postncomment[pd.notnull(df_postncomment['re_message.x'])]
rm_na.index = range(len(rm_na))
dfinal_fb = pd.DataFrame(
    rm_na,
    columns = ['key', 'created_time.x', 'id.x', 'message.x', 'id.y', 'message.y', 
               'lang.x', 're_message.x', 'lang.y', 're_message.y'])
dfinal_fb.to_csv(
    'final_facebook_preprocessing.csv',
    encoding = 'UTF-8',
    columns = ['key', 'created_time.x', 'id.x', 'message.x', 'id.y', 'message.y',
               'lang.x', 're_message.x', 'lang.y', 're_message.y'])
os.remove('facebook_preprocessing.csv')
#print(rm_na['re_message.x'][8])

test = pd.read_csv('final_facebook_preprocessing.csv', encoding = 'UTF-8', sep = ',', index_col = 0)
display(test.head(3))
print(len(test))



