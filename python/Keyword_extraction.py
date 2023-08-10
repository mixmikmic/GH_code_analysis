import json
import os
from pymongo import MongoClient
import codecs
import pandas as pd
import gensim
from gensim.models import phrases
from gensim.utils import lemmatize
import en_core_web_sm
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from sklearn.externals import joblib

client = MongoClient()
db = client.lingbuzz
papers = db.get_collection('papers')

def build_trigram_model(corpus):
    corpus = lemmer(nlp(corpus))
    bigram = phrases.Phrases(corpus, min_count=20, threshold=17)
    bigram_phraser = phrases.Phraser(bigram)
    trigram = phrases.Phrases(bigram_phraser[corpus], min_count=20, threshold=17)
    trigram_phraser = phrases.Phraser(trigram)
    return bigram_phraser, trigram_phraser

def punct_space(token):
    """
    helper function to eliminate punctuation, spaces and numbers.
    """
    return token.is_punct or token.is_space or token.like_num

#def remove_stopwords(tigrammized):
#    no_stop = [[term for term in sent if term not in my_stopwords] for sent in trigrammized]
 #   return no_stop
    
def remove_stopwords(stuff):
    # gives list of strings. Vectorizer needs this.
    out = []
    for sent in stuff:
        for term in sent:
            if term not in my_stopwords:
                out.append(term)
    return out

def remove_stopwords2(stuff):
    # gives list of list of strings. Phraser needs this.
    out = []
    for sent in stuff:
        out.append([term for term in sent if term not in my_stopwords])
    return out

def trigrammer(doc):
    tokens = nlp(doc)
    lemmas = lemmer(tokens)
    tokens_ = bigrams[lemmas]
    trigrammized = trigrams[tokens_]
    return [j for j in trigrammized]

def lemmer(tokens):
    """
    lemmatize words
    """
    word_space = []
    for sent in tokens.sents:
        sentence = []
        for token in sent:
            if not punct_space(token):
                if token.lemma_=='-PRON-':
                    sentence.append(token.lower_)
                else:
                    sentence.append(token.lemma_.strip('-'))
        word_space.append(sentence)
    return word_space

def my_tokenizer(doc):
    trigrammized = trigrammer(doc)
    return trigrammized

with open('lingbuzz.json') as f:
    meta_data = json.load(f)

meta_data[:4]

for di in meta_data:
        papers.update_one({'title': di['title']}, {'$set': {"abstract": di['abstract']}})

keywords = []
titles = []
authors = []

for doc in papers.find():
    keywords+=(doc['keywords'])
    titles.append(doc['title'])
    authors+=(doc['authors'])

abstracts = []
for doc in papers.find({'abstract':{'$exists': True}}):
    abstracts+=doc['abstract']

conclusions = []
for doc in papers.find({'paper':{'$exists': True}}):
    paper = doc['paper']
    to_match = doc['title']
    index_conclusion = paper.lower().rfind('conclusion')
    index_bibliography = paper.lower().rfind('bibliography')
    if index_conclusion != -1 and index_bibliography != -1:
        conclusion = paper[index_conclusion: index_bibliography]
        papers.update_one({'title': to_match}, {'$set': {'conclusion': conclusion }})
        conclusions.append(conclusion)

standard_stopwords = set(list(ENGLISH_STOP_WORDS)+list(stopwords.words('english')))

stopwords = joblib.load('../Aca_paper/notebooks/stopwords')

standard_stopwords

from string import digits
import re, string

# note: to get rid of unicode and wrongly tokenized words in the docs themselves, I have to find a way to get rid 
# of all words that contain punctuation except '-', and then substitute '-' by ''.

def eliminate_non_english_words(s):
    """takes list of words and eliminates all words that contain non-english characters, digits or punctuation"""
    english_words = []
    for word in s:
        try:
            word.encode(encoding='utf-8').decode('ascii')
            # if re.sub('-', '', word).isalpha():
                # english_words.append(re.sub('[%s]' % re.escape(string.punctuation), '', word))
            word = re.sub('[%s]' % re.escape(string.punctuation), '', word)
            if word.isalpha():
                english_words.append(word) 
        except UnicodeDecodeError:
            pass
    return english_words

def remove_standard_stopwords(li):
    """takes list of texts and returns list of words without standard stopwords, punctuation or digits"""
    words = []
    for text in li:
        no_weirdness = eliminate_non_english_words(text.split())
        for w in no_weirdness:
            if w.lower() not in standard_stopwords:
                words.append(w.lower())
        #words+= eliminate_non_english_words([w.lower() for w in text.split() if w.lower() not in standard_stopwords])
    return words

keywords_flat = remove_standard_stopwords([re.sub('[%s]' % re.escape(string.punctuation), '', w) for k in keywords for w in k.split()])

abstracts_flat = remove_standard_stopwords(abstracts)

'is' in keywords_flat

titles_flat = remove_standard_stopwords(titles)

authors

def flatten_authors(li):
    english_words = []
    for author in li:
        for word in author.split():
            english_words.append( re.sub('[%s]' % re.escape(string.punctuation), '', word))
    words = []
    for w in english_words:
        if len(w)>1:
            if w.lower() not in standard_stopwords:
                words.append(w.lower())
            #words+= eliminate_non_english_words([w.lower() for w in text.split() if w.lower() not in standard_stopwords])
    return words

# I will also flatten out the authors, to glue composite names together and seperate first names from last names. 
# In the text we will usually find the last name only.
authors_flat = flatten_authors(authors)

authors_flat

flat_conclusions = remove_standard_stopwords(conclusions)
## the conclusions contain too much trash, I won't extract keywords from them.

top_k_words = set(titles_flat + abstracts_flat + keywords_flat + authors_flat + ['v2'])

len(top_k_words)

joblib.dump(top_k_words, 'top_k_words')

'phonology)?' in top_k_words

top_k_words

joblib.dump(authors_flat, 'authors')



