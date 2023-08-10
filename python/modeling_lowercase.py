import os
import re
import json
from pprint import pprint
from bs4 import BeautifulSoup
from bs4.element import Comment

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

# Raghu's code to pull down jobs
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

print('Number of all job titles:', len(all_job_titles))
uniq_job_titles = set(all_job_titles)
print('Number of unique job titles:', len(uniq_job_titles))

print('Number of all job ids:', len(all_job_ids))
uniq_job_ids = set(all_job_ids)
print('Number of unique job ids:', len(uniq_job_ids))

uniq_job_ids = set(all_job_ids)
print('Number of unique job ids:', len(uniq_job_ids))

print('\nNumber of summaries:',len(all_summaries))
unique_summaries = set(all_summaries)
print('Number of unique summaries:', len(unique_summaries))

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def make_text(text):
    soup = BeautifulSoup(text, "html5lib")
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip().lower() for t in visible_texts) # Modified 27MAR18 to produce lowercase results

get_ipython().run_cell_magic('time', '', 'texts = []\n\nfor item in unique_summaries:\n    texts.append(make_text(item))')

get_ipython().run_cell_magic('time', '', "# write to disk as a single large file\nwith open('../additional_jobs/s3_visible_texts_02apr.txt', 'w') as outfile:\n    for item in texts:\n        outfile.write(item)\n        outfile.write('\\n')")

texts = []

with open('../additional_jobs/s3_visible_texts_02apr.txt', 'r') as infile:
    for line in infile:
        line = line.strip()
        texts.append(line)

# Optional preprocessing step... maybe see if this works later

# remove words that appear only once
# texts = [[word for word in document.lower().split() if word not in stop_words]
#          for document in parsed_dataset]

# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# texts = [[token for token in text if frequency[token] > 1] for text in texts]

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    SRG: modified for a list
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    for review in filename:
        yield review.replace('\\n', '\n')
            
def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

get_ipython().run_cell_magic('time', '', "\nwith codecs.open('../additional_jobs/spacy_jds_concat_parsed.txt', 'w', encoding='utf_8') as f:\n    for sentence in lemmatized_sentence_corpus(texts): # Changed 28MAR18 to include tokenization function call\n        f.write(sentence + '\\n')")

unigram_sentences = LineSentence('../additional_jobs/spacy_jds_concat_parsed.txt')

# print some examples
for unigram_sentence in it.islice(unigram_sentences, 230, 240):
    print(u' '.join(unigram_sentence))
    print(u'')

get_ipython().run_cell_magic('time', '', "\nbigram_model = Phrases(unigram_sentences)\nbigram_model.save('../additional_jobs/spacy_bigram_model_all_PARSED')")

# load the finished model from disk
bigram_model = Phrases.load('../additional_jobs/spacy_bigram_model_all_PARSED')

get_ipython().run_cell_magic('time', '', "\nwith codecs.open('../additional_jobs/spacy_bigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n    for unigram_sentence in unigram_sentences:\n        bigram_sentence = u' '.join(bigram_model[unigram_sentence])\n        f.write(bigram_sentence + '\\n')")

bigram_sentences = LineSentence('../additional_jobs/spacy_bigram_sentences_PARSED.txt')

# # print examples; certain bigrams are underlined
# for bigram_sentence in it.islice(bigram_sentences, 240, 250):
#     print(u' '.join(bigram_sentence))
#     print(u'')

get_ipython().run_cell_magic('time', '', "trigram_model = Phrases(bigram_sentences)\ntrigram_model.save('../additional_jobs/spacy_trigram_model_all_PARSED')")

# load the finished model from disk
trigram_model = Phrases.load('../additional_jobs/spacy_trigram_model_all_PARSED')

get_ipython().run_cell_magic('time', '', "\nwith codecs.open('../additional_jobs/spacy_trigram_sentences_PARSED.txt', 'w', encoding='utf_8') as f:\n    for bigram_sentence in bigram_sentences:\n        trigram_sentence = u' '.join(trigram_model[bigram_sentence])\n        f.write(trigram_sentence + '\\n')")

trigram_sentences = LineSentence('../additional_jobs/spacy_trigram_sentences_PARSED.txt')

for trigram_sentence in it.islice(trigram_sentences, 240, 250):
    print(u' '.join(trigram_sentence))
    print(u'')

# NLTK stopwords format is not a list; use this version
stopwords = get_stop_words('english')
additional_stopwords = ["\'s", 'or']
stopwords = stopwords + additional_stopwords
print(stopwords)

get_ipython().run_cell_magic('time', '', "\nwith codecs.open('../additional_jobs/spacy_trigram_transformed_jds_all_PARSED.txt', 'w', encoding='utf_8') as f:\n    for parsed_review in nlp.pipe(line_review(texts),\n                                  batch_size=10000, n_threads=24):\n        # lemmatize the text, removing punctuation and whitespace\n        unigram_review = [token.lemma_ for token in parsed_review\n                          if not punct_space(token)]\n\n        # apply the first-order and second-order phrase models\n        bigram_review = bigram_model[unigram_review]\n        trigram_review = trigram_model[bigram_review]\n\n        # remove any remaining stopwords\n        trigram_review = [term for term in trigram_review\n                          if term not in stopwords]\n\n        # write the transformed review as a line in the new file\n        trigram_review = u' '.join(trigram_review)\n        f.write(trigram_review + '\\n')")

# probably another memory issue.

get_ipython().run_cell_magic('time', '', "\ntrigram_reviews = LineSentence('../additional_jobs/spacy_trigram_sentences_PARSED.txt')\n\n# learn the dictionary by iterating over all of the reviews\ntrigram_dictionary = Dictionary(trigram_reviews)\n\n# filter tokens that are very rare or too common from\n# the dictionary (filter_extremes) and reassign integer ids (compactify)\ntrigram_dictionary.filter_extremes(no_below=10, no_above=0.4)\ntrigram_dictionary.compactify()\n\ntrigram_dictionary.save('../additional_jobs/spacy_trigram_dict_all.dict')\n    \n# load the finished dictionary from disk\ntrigram_dictionary = Dictionary.load('../additional_jobs/spacy_trigram_dict_all.dict')")

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """    
    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

get_ipython().run_cell_magic('time', '', "\n# generate bag-of-words representations for all JDs and save them as a matrix\nMmCorpus.serialize('../additional_jobs/spacy_trigram_bow_corpus_all.mm',\n                   trigram_bow_generator('../additional_jobs/spacy_trigram_sentences_PARSED.txt'))")

