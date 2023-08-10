import logging

# utilities and plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# node2vec stuff
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO)

network_emb = Word2Vec.load('data/sosweet-network/undir_weighted_mention_network_thresh_5.emb')
lang_emb = Word2Vec.load('data/sosweet-w2v/dim_50/lowe_dim_sosweet2vec.w2v')

len(network_emb.wv.vocab)

len(lang_emb.wv.vocab)

import csv
import gzip
from collections import defaultdict
import re
import nltk

url_regex = re.compile('http[^ ]')

def words(text):
    # Remove urls (which contain : and / characters)
    re.sub(url_regex, '', text)
    # TODO: transliterate accents
    # TODO: split on ?!;:,.<>'"`=+-*%$/\|()
    # TODO: remove stuff starting with @, and 'RT'

user_words = defaultdict(set)
with gzip.open('data/sosweet-text/undir_weighted_mention_network_thresh_5/'
               '2017-10-pipe-user_timestamp_body-csv-pipe-filter_users.gz', 'rt') as tweets_file:
    reader = csv.DictReader(tweets_file, fieldnames=['user_id', 'timestamp', 'body'])
    for i, tweet in enumerate(reader):
        user_words[int(tweet['user_id'])].update(words(tweet['body']))

import unicodedata

lang_emb.wv[tweet['body'][-1]]

unicodedata.category('界')



