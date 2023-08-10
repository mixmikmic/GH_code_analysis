import pandas as pd
import sqlite3
import gensim
import nltk
import glob
import json
import pickle
from tqdm import tqdm_notebook as tn

## Helpers

def save_pkl(target_object, filename):
    with open(filename, "wb") as file:
        pickle.dump(target_object, file)
        
def load_pkl(filename):
    return pickle.load(open(filename, "rb"))

def save_json(target_object, filename):
    with open(filename, 'w') as file:
        json.dump(target_object, file)
        
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Loading metadata from trainning database
con = sqlite3.connect("F:/FMR/data.sqlite")
db_documents = pd.read_sql_query("SELECT * from documents", con)
db_authors = pd.read_sql_query("SELECT * from authors", con)
data = db_documents # just a handy alias
data.head()

import spacy
nlp = spacy.load('en')

def get_name(s):
    end = 0
    for i in range(len(s.split('/')[0])):
        try:
            a = int(s[i])
            end = i
            break
        except:
            continue
    return s[:end]

journals = []
for i in db_documents['submission_path']:
    journals.append(get_name(i))

journals = set(journals)

from gensim.models.phrases import Phraser, Phrases

from itertools import tee
import multiprocessing

# Use tn(iter, desc="Some text") to track progress
def gen_tokenized_dict_beta(untokenized_dict):
    gen1, gen2 = tee(untokenised.items())
    ids = (id_ for (id_, text) in gen1)
    texts = (text for (id_, text) in gen2)
    docs = nlp.pipe(tn(texts, desc="Tokenization", total=len(untokenized_dict)), n_threads=9)
    tokenised = {id_: doc for id_, doc in zip(ids, docs)}
    return tokenised

def gen_tokenized_dict(untokenized_dict):
    return {k: nlp(v) for k, v in tn(untokenized_dict.items(), desc="Tokenization")}

def gen_tokenized_dict_parallel(untokenized_dict): # Uses textblob
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as executor:
         return {num:sqr for num, sqr in tn(zip(untokenized_dict.keys(), executor.map(TextBlob, untokenized_dict.values())), desc="Tokenization")}

def keep_journal(dict_, journal):
    kept = {k: v for k, v in tn(dict_.items(), desc="Journal Filter") if k.startswith(journal)}
    print("Original: ", len(dict_), ", Kept ", len(kept), " items.")
    return kept

import os
from spacy.tokens.doc import Doc
def save_doc_dict(d, folder_name):
    os.mkdir(folder_name)
    nlp.vocab.dump_vectors(os.path.join(folder_name, 'vocab.bin'))
    for k, v in tn(d.items(), desc="Saving doc"):
        k = k.replace('/', '-') + '.doc'
        with open(os.path.join(folder_name, k), 'wb') as f:
            f.write(v.to_bytes())
            
def load_doc_dict(folder_name):
    nlp = spacy.load('en') # This is very important
    file_list = glob.glob(os.path.join(folder_name, "*.doc"))
    d = {}
    nlp.vocab.load_vectors_from_bin_loc(os.path.join(folder_name, 'vocab.bin'))
    for k in tn(file_list, desc="Loading doc"):
        with open(os.path.join(k), 'rb') as f:
            k_ = k.split('\\')[-1].replace('-', '/').replace('.doc', '')
            for bs in Doc.read_bytes(f):
                d[k_] = Doc(nlp.vocab).from_bytes(bs)
    return d

def pos_filter(l, pos="NOUN"):
    return [str(i.lemma_).lower() for i in l if i.pos_ == 'NOUN' and i.is_alpha]

def bigram(corpus):
    phrases = Phrases(corpus)
    make_bigram = Phraser(phrases)
    return [make_bigram[i] for i in tn(corpus, desc='Bigram')]

# Set training parameters.
num_topics = 150
chunksize = 2000
passes = 1
iterations = 150
eval_every = None  # Don't evaluate model perplexity, takes too much time.

models = {}
vis_dict = {}

import gensim.corpora
import pyLDAvis.gensim
import warnings
from imp import reload
warnings.filterwarnings("ignore")
def train_journal(j):
    corpus = load_doc_dict(j)
    corpus = {k: pos_filter(v) for k, v in tn(corpus.items())}
    
    # Make it bigram
    
    tokenised_list = bigram([i for i in corpus.values()])
    # Create a dictionary for all the documents. This might take a while.
    reload(gensim.corpora)
    print(tokenised_list[0][:10])
    dictionary = gensim.corpora.Dictionary(tokenised_list)
    # dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=None)
    if len(dictionary) < 10:
        print("Warning: dictionary only has " + str(len(dictionary)) + " items. Passing.")
        return None, None
    corpus = [dictionary.doc2bow(l) for l in tokenised_list]
    # Save it for future usage
    from gensim.corpora.mmcorpus import MmCorpus
    MmCorpus.serialize(os.path.join(j, "noun_bigram.mm"), corpus)
    # Also save the dictionary
    dictionary.save(os.path.join(j, "_noun_bigram.ldamodel.dictionary"))
    # Train LDA model.
    from gensim.models import LdaModel
    
    # Train LDA model
    print(len(dictionary))
    # Make a index to word dictionary.
    print("Dictionary test: " + dictionary[0])  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize,                            alpha='auto', eta='auto',                            iterations=iterations, num_topics=num_topics,                            passes=passes, eval_every=eval_every)
    model.save(os.path.join(j, "_noun_bigram_" + str(num_topics) + ".ldamodel"))
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    del dictionary
    return model, vis

journals = set([i for i in journals if i])
for j in tn(journals, desc="Journal"):
    try:
        if j in models:
            print(j, 'already exists. Skipping.')
            continue
        model, vis = train_journal(j)
        if model and vis:
            models[j] = model
            vis_dict[j] = vis
            save_pkl(filename=os.path.join(j, '_bigram_vis.pkl'), target_object=vis)
    except Exception as e:
        print(e)

len(models)

import pyLDAvis

pyLDAvis.display(vis_dict['pacis'])

journals

