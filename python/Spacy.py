import os
import sys
import logging
import warnings
import copy
import re
import json
import itertools
from operator import attrgetter
import inspect
import collections
import numpy as np
import pandas as pd

import spacy
from spacy import displacy
from spacy.lang.en import English
import en_core_web_sm

from IPython.display import display, HTML

warnings.filterwarnings('ignore')
np.random.seed(42)

def as_table(obj_iter, variables):
    dct = { var:[] for var in variables}
    for obj in obj_iter:
        for var in variables:
            attr = attrgetter(var)(obj)
            if (not isinstance(attr, str)) and isinstance(attr, collections.Iterable):
                attr = list(attr)
            dct[var].append(attr)
    return pd.DataFrame(data=dct, columns=variables)

# Load the english model
# Disabling the 'tagger' will result in different lemmatization
nlp = spacy.load('en')
# Alternatives
# nlp = spacy.load('en', disable=['parser', 'ner'])
# nlp = en_core_web_sm.load()
# nlp = English()
print('NLP pipeline: ', nlp.pipe_names)

doc = nlp(u"The company Apple is looking at buying U.K. startups for $1 billion. I'm not.")

print('doc: ', doc)

df_tokens = as_table(
    doc, ['text', 'lemma_', 'pos_', 'norm_', 'tag_', 'dep_', 'children', 'shape_', 'is_alpha', 'is_stop'])
display(df_tokens)
print([t.lemma_ for t in doc])

displacy.render(doc, style='dep', jupyter=True)

df_ents = as_table(
    doc.ents, ['text', 'start_char', 'end_char', 'label_'])
display(df_ents)

displacy.render(doc, style='ent', jupyter=True)

df_np = as_table(
    doc.noun_chunks, ['text', 'root.text', 'root.dep_', 'root.head.text'])
display(df_np)

df_sent = as_table(
    doc.sents, ['text'])
display(df_sent)

df_sent = as_table(
    (word for sent in doc.sents for word in sent), ['text', 'orth_', 'tag_', 'head.i', 'dep_'])
display(df_sent)

print(spacy.about.__version__)
s = u"The company Apple is looking at buying U.K. startups for $1 billion. I'm not."

nlp = spacy.load('en', disable=['parser', 'ner'])
doc = nlp(s)
print('With tagger: ', [t.lemma_ for t in doc])  # Lemma's are lowercase

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
doc = nlp(s)
print('Without tagger: ', [t.lemma_ for t in doc])  # Lemma's captial letters are kept

print(spacy.about.__version__)
nlp = spacy.load('en')
for s in ["The store", "the store"]:
    doc = nlp(s)
    print('\n{}'.format(s))
    for t in doc:
        print('{}\t{}'.format(t.text, t.is_stop))

from spacy.lang.en.stop_words import STOP_WORDS
print(STOP_WORDS)

