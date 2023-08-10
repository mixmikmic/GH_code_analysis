import tweepy
consumer_key = "51MI8RrYmzO4btCKG4Qb5uqAa"
consumer_secret = "ajpPv3Ag0NvMEQLBIwiPyDyU78BbLZn8IS1gTba4x9ZOHNPMNM"
access_token = "3004471069-VDbNpT9NO0QOtiqKZXkoH5Flv4MArCflIYImXjn"
access_token_secret = "sP6KMjPZXxYAnaae8bOiauLjCVnx8bzWkBk4KU1iZBxdl"

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth) 

n = 3 # Tweet count

# Print tweets from my home timeline
for status in tweepy.Cursor(api.home_timeline).items(n):
    # Process a single status
    print(status.text)
#     print(status._json)

# Getting tweets from the New York Times user
for tweet in api.user_timeline(id="nytimes", count=n):
    # printing the text stored inside the tweet object
    print(tweet.text)

# get tweets by keyword
for tweet in api.search(q="stock", lang="en")[:n]:
    # printing the text stored inside the tweet object
    print(tweet.user.screen_name,"Tweeted:",tweet.text)

from nltk.tokenize import word_tokenize

# note a default tokenizer doesn't do a good job
tweets = api.search(q="stock", lang="en")
tweet  = tweets[0].text
print(word_tokenize(tweet))

import re
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def chunk(s):
    return tokens_re.findall(s)
 
def tokenize(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

print(tokenize(tweet.text))
# ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']

from nltk.corpus import stopwords
import string
 
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

def filt(terms, stop):
    return [term for term in terms if term not in stop]

import operator 
import json
from collections import Counter
 
count_all = Counter()
for tweet in tweets[:3]:
    # Create a list with all the terms
    terms_all = filt(tokenize(tweet.text), stop)
    # Update the counter
    count_all.update(terms_all)
    # Print the first 5 most frequent words
    print(tweet.text)
    print(count_all.most_common(5))

import operator 
import json
from collections import Counter
 
count_all = Counter()
for tweet in tweets[:3]:
    # Create a list with all the terms
    terms_all = tokenize(tweet.text)
    # Update the counter
    count_all.update(terms_all)
    # Print the first 5 most frequent words
    print(tweet.text)
    print(count_all.most_common(5))

# # Count terms only once, equivalent to Document Frequency
# terms_single = set(terms_all)
# # Count hashtags only
# terms_hash = [term for term in preprocess(tweet['text']) 
#               if term.startswith('#')]
# # Count terms only (no hashtags, no mentions)
# terms_only = [term for term in preprocess(tweet['text']) 
#               if term not in stop and
#               not term.startswith(('#', '@'))] 
#               # mind the ((double brackets))
#               # startswith() takes a tuple (not a list) if 
#               # we pass a list of inputs

from nltk import bigrams

terms_bigram = bigrams(stop)

list(terms_bigram)



