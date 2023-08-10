import os
import re
import json
from pprint import pprint
from bs4 import BeautifulSoup
from bs4.element import Comment
import sys

import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import gensim
from pprint import pprint

from stop_words import get_stop_words

import codecs
import os
import time

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')

import spacy
import pandas as pd
import itertools as it
import en_core_web_sm

import spacy
nlp = spacy.load('en')

import codecs
import boto3
# from collections import defaultdict
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LsiModel

import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle

s3 = boto3.resource('s3')

BUCKET_NAME = 'tech-salary-project'

# %%time

# bucket = s3.Bucket(BUCKET_NAME)
# all_job_titles = []
# all_job_ids = []
# all_summaries = []
# for o in bucket.objects.all():
#     if o.key.startswith('salaries'):
#         continue
    
#     object = bucket.Object(o.key)
#     try:
#         lines = object.get()['Body'].read().decode('utf-8').splitlines()
#         for line in lines:
#             d = json.loads(line)
            
#             title = d['title']
#             jid = d['id']
#             summary = d['summary']
            
#             all_job_ids.append(jid)
#             all_job_titles.append(title)
#             all_summaries.append(summary)
#     except Exception as e:
#         continue

print(len(set(all_summaries)))

get_ipython().run_cell_magic('time', '', "\n# Put dataset in memory\ndirectory = os.fsencode('../local_data/')\n\noriginal_texts = []\n\nfor file in os.listdir(directory):\n    filename = os.fsdecode(file)\n    full_file = '../local_data/' + str(filename)\n    with open(full_file, 'r') as infile:\n        text = infile.read()\n        original_texts.append(text)")

get_ipython().run_cell_magic('time', '', "# Load in Stackoverflow data\n\nstackoverflow_raw = []\n\ndef clean_text(text):\n    cleanr = re.compile(r'\\bamp\\b')\n    cleantext = re.sub(cleanr, '', text)\n    return cleantext\n\nwith open('../stackoverflow_data/stack_parsed.txt', 'r') as infile:\n    for line in infile:\n        line = line.strip()\n        line = clean_text(line)\n        if line == '':\n            continue\n        stackoverflow_raw.append(line)\n\n# Size of list in bytes\nprint(sys.getsizeof(stackoverflow_raw))\nprint(len(stackoverflow_raw))")

print("Length of job data list:", len(original_texts))
print("Length of Stackoverflow data list:", len(stackoverflow_raw))
all_data = original_texts + stackoverflow_raw
print("Combined dataset length:", len(all_data))

print("Size of total dataset in MB:", sys.getsizeof(all_data)/1000000)

get_ipython().run_cell_magic('time', '', 'lower_stackoverflow = [item.lower().strip() for item in stackoverflow_raw]')

# subset = lower_data[::10000]
# print("Items in subset:",len(subset))

# Load pickled skill dictionary from disk. I generated it from prior work.
with open('skill_bigram_dict.pkl', 'rb') as handle:
    skill_dict = pickle.load(handle)
print("machine learning becomes", skill_dict['machine learning'])

def multiple_replace(dict, text):
    '''
    Function that uses regex to replace terms.
    Input is a dictionary of terms to switch, plus
    a string.
    '''
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

get_ipython().run_cell_magic('time', '', '# new_data = []\n# for text in lower_data:\n#     new_data.append(multiple_replace(skill_dict, text))\n\nnew_stackoverflow = []\nfor text in lower_stackoverflow:\n    new_stackoverflow.append(multiple_replace(skill_dict, text))')

# Write list to file as a pickled object, to save time later
with open('../models_12apr/stackoverflow_data_skill_bigrams_list.pkl', 'wb') as f:
    pickle.dump(new_stackoverflow, f)

# Load back into memory
with open('../models_12apr/all_lowercase_data_skill_bigrams_list.pkl', 'rb') as handle:
    new_data = pickle.load(handle)

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    12APR: adding numbers to this
    """
    
    return token.is_punct or token.is_space or token.is_digit

def line_review(filename):
    """
    SRG: modified for a list
    generator function to read in text from the file
    and un-escape the original line breaks in the text
    """
    
    for text in filename:
        yield text.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_text in nlp.pipe(line_review(filename),
                                  batch_size=5000, n_threads=24):
        
        for sent in parsed_text.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

get_ipython().run_cell_magic('time', '', "\nwith codecs.open('../models_12apr/jds_skill_bigrams_concat_parsed_stackoverflow.txt', 'w', encoding='utf_8') as f:\n    for sentence in lemmatized_sentence_corpus(new_stackoverflow):\n        f.write(sentence + '\\n')")

