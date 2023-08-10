import spacy
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, VectorizerMixin
from utils import *

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth', 120)

# Note: you can add other languages that Spacy supports, or download
# larger models for english that Spacy offers. 
nlp = spacy.load('en') 

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def stop_word_removal(li):    
    return [l for l in li if l not in ENGLISH_STOP_WORDS]

from utils import clean_html
from sklearn.feature_extraction.text import strip_accents_unicode


def clean_twitter(s):
    """ Cleans Twitter specific issues 
    
    Can you think of what else you might need to add here?
    """
    s = sub(r'@\w+', '', s) #remove @ mentions from tweets    
    return s

def preprocessor(s):
    """ For all basic string cleanup. 
    
    Think of what you can add to this to improve things. What is
    specific to your goal, how can you transform the text. Add tokens,
    remove things, unify things. 
    """
    s = clean_html(s)
    s = strip_accents_unicode(s.lower())
    s = clean_twitter(s)
    return s

import spacy

def cool_tokenizer(sent):
    """ Idea from Travis in class: adds a token to the end with nsubj and root verb!"""
    doc = nlp(sent)
    tokens = sorted(doc, key = lambda t: t.dep_)
    return ' '.join([t.lemma_ for t in tokens if t.dep_ in ['nsubj', 'ROOT']])

cool_tokenizer('a migrant died in crossing the river')

from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect

def dep_tokenizer(sent):
    """ A simple version of tokenzing with the dependencies.
    
    Note: this is monolingual! Also, it still doesn't take into 
    account correlations!
    """
    doc = nlp(sent)
    tokens = [t for t in doc if not t.is_stop and t.dep_ not in ['punct', '']]
    return [':'.join([t.lemma_,t.dep_]) for t in tokens]

dep_tokenizer('a migrant died in crossing the river')

import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

def analyzer(s, ngram_range = (1,2)):
    """ Does everything to turn raw documents into tokens.  
    
    Note: None of the above tokenizers are implemented!
    """
    s = preprocessor(s)
    pattern = re.compile(r"(?u)\b\w\w+\b")
    unigrams = pattern.findall(s)
    unigrams = [u for u in unigrams if u not in ENGLISH_STOP_WORDS]
    tokens = ngrammer(unigrams, ngram_range)
    return tokens

X = pd.read_csv('kaggle/train.csv').tweet
y = pd.read_csv('kaggle/train.csv').label

cutoff = 1750
X_train, X_test, y_train, y_test = X[0:cutoff], X[cutoff:], y[0:cutoff], y[cutoff:]

X_test.shape, y_test.shape

def create_vectors(X_train, X_test, analyzer = analyzer):
    """ Just a small helper function that applies the SKLearn Vectorizer with our analyzer """
    idx = X_train.shape[0]
    X = pd.concat([X_train, X_test])
    vectorizer = TfidfVectorizer(analyzer=analyzer).fit(X)
    vector = vectorizer.transform(X)
    return vector[0:idx], vector[idx:], vectorizer

V_train, V_test, vectorizer = create_vectors(X_train, X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score

model = MultinomialNB(class_prior=[0.5,0.5])
model.fit(V_train, y_train)
preds = model.predict_proba(V_test)[:,1]
roc_auc_score(y_test, preds)

from sklearn.svm import LinearSVC

model = LinearSVC(tol = 10e-6, max_iter = 8000)
model.fit(V_train, y_train)
preds = model.decision_function(V_test)
roc_auc_score(y_test, preds)

# Look at your false predictions!
false_pos, false_neg = get_errors(X_test, y_test, preds)

test_df = pd.read_csv('kaggle/test.csv')
X_sub, id_sub = test_df.tweet, test_df.id
V_train, V_test, _ = create_vectors(X, X_sub)
model.fit(V_train, y)
preds = model.decision_function(V_test)

write_submission_csv(preds, id_sub, 'kaggle/submission.csv')

