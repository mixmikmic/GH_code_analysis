from collections import OrderedDict
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('../')
import config
import pymysql.cursors
import pandas as pd
import spacy

nlp = spacy.load('en')

connection = pymysql.connect(host='localhost',
                             user='root',
                             password=config.MYSQL_SERVER_PASSWORD,
                             db='youtubeProjectDB',
                             charset='utf8mb4', 
                             cursorclass=pymysql.cursors.DictCursor)


mypath = '../textbooks'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
with connection.cursor() as cursor:
                       
            sql = """
            SELECT search_api.videoId, videoTitle, captionsText, wordCount, captions.id 
            FROM search_api
            INNER JOIN captions
            ON search_api.videoId = captions.videoId
            WHERE captions.id 
            IN (5830, 45, 52, 54, 6195, 6198, 6203, 6208, 14525, 14523, 14518);"""            
            cursor.execute(sql)
            manyCaptions = cursor.fetchall()
            videos_df = pd.read_sql(sql, connection)
                        
connection.close()

L1 = []
L2 = []
for file in onlyfiles:
    L1.append((file ,  (open(mypath + '/' + file, 'r').read()) ))
TextBooksDict = OrderedDict(L1)

for item in manyCaptions:
    #  L2.append((item.get('id')  ,  item.get('captionsText')))  # 'id' key is lower case!!!
    L2.append((item.get('videoTitle')  ,  item.get('captionsText')))
ManyCaptionsDict = OrderedDict(L2)   

# Merge OrderedDict's'
L3 = []
for k, v in zip(ManyCaptionsDict.keys(), ManyCaptionsDict.values()):
    L3.append((k,v))
for k, v in zip(TextBooksDict.keys(), TextBooksDict.values()):
    L3.append((k,v))
UnitedOrderedDict = OrderedDict(L3)

videos_df['characterCount'] = videos_df['captionsText'].map(len)
# reorder the columns
videos_df['charPerWord'] = videos_df.characterCount / videos_df.wordCount
videos_df = videos_df.reindex(columns=['videoTitle','characterCount','wordCount', 'charPerWord','captionsText','id', 'videoId'])

textbooks_df = pd.read_pickle('textbooksDF.pickle') 

# NB - this cell can take minutes to run rather load from pickle if nothing added. 
# https://chrisalbon.com/python/pandas_create_column_with_loop.html
fileName = [k for k in TextBooksDict.keys()]
characterCount = [len(TextBooksDict.get(k)) for k in TextBooksDict.keys()]
wordCount = [len(nlp(TextBooksDict.get(k))) for k in TextBooksDict.keys()]
raw_data = {'fileName' : fileName,
            'characterCount': characterCount,
            'wordCount':wordCount}
textbooks_df = pd.DataFrame(raw_data, columns = ['fileName', 'characterCount', 'wordCount'])
textbooks_df['charPerWord'] = textbooks_df.characterCount / textbooks_df.wordCount

textbooks_df.to_pickle('textbooksDF.pickle') 

videos_df[['videoTitle', 'characterCount', 'wordCount', 'charPerWord']].head(5)

textbooks_df.head(5)

print (textbooks_df.wordCount.mean())
print (videos_df.wordCount.mean())

documents = [TextBooksDict.get(key) for key in list(TextBooksDict)]
# following two rows are used in a pretty-print thing at the bottom of the notebook
# to put the labels back on to an otherwise unlabeled NumPy array
row_labels = list(TextBooksDict)
column_labels = list(TextBooksDict)

documents = [ManyCaptionsDict.get(key) for key in list(ManyCaptionsDict)]

row_labels = list(ManyCaptionsDict)
column_labels = list(ManyCaptionsDict)

documents = [UnitedOrderedDict.get(key) for key in list(UnitedOrderedDict)]

row_labels = list(UnitedOrderedDict)
column_labels = list(UnitedOrderedDict)

from spacy.en import English
parser = English()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()

    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))   
# (containing the SpacY tokenizer tokenizeText)

# the pipeline to clean, tokenize, vectorize, and classify
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])

p = pipe.fit_transform(documents)   # takes 20 min with ~10 textbooks!

pairwise_similarity = (p * p.T).A #  In Scipy, .A transforms a sparse matrix to a dense one

df9 = pd.DataFrame(pairwise_similarity, columns=column_labels, index=row_labels)
df9.head(3)

import numpy as np # save the 15 minutesfromlasttime
#np.save('pairwise_similarity_35textsAndCaptions', pairwise_similarity)   
#np.save('35textLabels', row_labels) 

pairwise_similarity[0]

(-pairwise_similarity[0]).argsort()

(-pairwise_similarity[0]).argsort()[1:4]

row_labels[0]
# http://stackoverflow.com/questions/18272160/access-multiple-elements-of-list-knowing-their-index

(np.array(row_labels))[((-pairwise_similarity[0]).argsort()[1:4])]

for i in range(len(row_labels)):
    print (row_labels[i], '\n', (np.array(row_labels))[((-pairwise_similarity[i]).argsort()[1:4])], '\n')

print("----------------------------------------------------------------------------------------------")
print("The original data as it appeared to the classifier after tokenizing, lemmatizing, stoplisting, etc")

transform = p 

# get the features that the vectorizer learned (its vocabulary)
vocab = vectorizer.get_feature_names()

# the values from the vectorizer transformed data (each item is a row,column index with value as # times occuring in the sample, stored as a sparse matrix)
for i in range(len(documents)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))
    print("Sample {}: {}".format(i, s))

