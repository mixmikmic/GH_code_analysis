import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

eng_stop = set(stopwords.words('english'))

"""
Data fetched from
https://www.kaggle.com/crowdflower/twitter-airline-sentiment/data
"""
df = pd.read_csv("Tweets.csv")
df.shape

def get_tokens(text):
    return word_tokenize(text)

def convert_lowercase(text):
    return text.lower()

def remove_punctuations(text):
    return re.sub(r'\W+', '', text)

def remove_stopwords(text):
    return [word for word in text.split() if word not in eng_stop]

### Lower casing ###
st = "This is natural language processing"
st = convert_lowercase(st)
print "Lower cased text : ", st

print "Tokens are : ", get_tokens(st)

print "After stopword removal : ", remove_stopwords(st)

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

### Stemming ###
porter_stemmer = PorterStemmer()

word = 'churches' #['maximum', 'multiply', 'presumably']
porter_stemmer.stem(word)

wordnet_lemmatizer = WordNetLemmatizer()

word = 'churches'
wordnet_lemmatizer.lemmatize(word)

