from tf_custom_models import OneLayerNN, SoftmaxClassifier
from utility import train_and_eval_auc, HATEBASE_FIELDS
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import roc_auc_score as AUC

import matplotlib.pyplot as plt

import os
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import json
import itertools

DATA_DIR = "data/hatebase"
EMBEDDING_SIZE = 100
STATE_SIZE = 100
GLOVE_SIZE = 1193514
GLOVE_PATH = "data/glove/glove.twitter.27B.%dd.txt" % EMBEDDING_SIZE

EMBED_PATH = "data/hatebase/embeddings.%dd.dat" % EMBEDDING_SIZE
HIDDEN_EMBED_PATH = "data/hatebase/embeddings.hidden.%dd.dat" % EMBEDDING_SIZE
HB_PATH = "data/hatebase/lexicon.csv"
VOCAB_PATH = "data/hatebase/vocab.dat"

def load_embeddings(embed_path, vocab, force=False):
    if not os.path.exists(embed_path):
        hb_vecs = np.zeros((len(vocab), EMBEDDING_SIZE))
        with open(GLOVE_PATH, 'r') as fh:
            found = []
            for line in tqdm(fh, total=GLOVE_SIZE):
                array = line.strip().split(" ")
                word = array[0]
                if word in vocab:
                    idx = vocab[word]
                    found.append(idx)
                    vector = list(map(float, array[1:]))
                    hb_vecs[idx, :] = vector
            # words not found are set to random values
            unfound = list(set(vocab.values()) - set(found))
            for i in unfound:
                hb_vecs[i, :] = np.random.randn(EMBEDDING_SIZE)
                
        hb_vecs = pd.DataFrame(hb_vecs)
        hb_vecs.to_csv(embed_path, header = False, index = False)
        return hb_vecs

    with open(embed_path, 'rb') as embed_path:
        data_x = pd.read_csv( embed_path, header = None, quoting = 0, dtype = np.float32 )
        return data_x

# grab the data
hatebase_data = pd.read_csv( HB_PATH, header = 0, index_col = 0, quoting = 0, 
                                dtype = HATEBASE_FIELDS, usecols = range(9) )
vocab = dict([(x, y) for (y, x) in enumerate(hatebase_data.index)])
hatebase_embeddings = load_embeddings(EMBED_PATH, vocab, True)

train_i, test_i = train_test_split( np.arange( len( hatebase_embeddings )), train_size = 0.8, random_state = 44 )
train_x = hatebase_embeddings.iloc[train_i]
test_x = hatebase_embeddings.iloc[test_i]
train_y = hatebase_data.iloc[train_i]
test_y = hatebase_data.iloc[test_i]

def total_jaccard( train_x, train_y, test_x, test_y, model ):
    model.fit( train_x, train_y )
    p = model.predict( test_x )
    #print p
    p = (p >= 0.5).astype(float)
    total = sum([jaccard_similarity_score(y_true, y_pred) for y_true, y_pred in zip(test_y, p)])
    print "Total Jaccard similarity:", total/len(test_x)

def train_and_eval_auc( train_x, train_y, test_x, test_y, model ):
    model.fit( train_x, train_y )
    p = model.predict_proba( test_x )
    p = p[:,1] if p.shape[1] > 1 else p[:,0]

    auc = AUC( test_y, p )
    print "AUC:", auc

tf.reset_default_graph()
nn = OneLayerNN(h=100)
nn.fit( hatebase_embeddings, hatebase_data )
hidden_states = nn.return_hidden_states( hatebase_embeddings )

# write hidden states
hidden_states = pd.DataFrame(hidden_states)
hidden_states.to_csv(HIDDEN_EMBED_PATH, header = False, index = False)

with open(VOCAB_PATH, mode="wb") as vocab_file:
    for w in hatebase_data.index.values:
        vocab_file.write(w + b"\n")

tf.reset_default_graph()
nn = OneLayerNN()
total_jaccard( train_x, train_y.iloc[:,:7], test_x, test_y.iloc[:,:7].values, nn )

for i, field in enumerate(HATEBASE_FIELDS):
    print field
    tf.reset_default_graph()
    train_and_eval_auc( train_x, train_y.iloc[:,i], test_x, test_y.iloc[:,i], OneLayerNN() )

lr = SoftmaxClassifier()

total_jaccard( train_x, train_y.iloc[:,:7], test_x, test_y.iloc[:,:7].values, lr )

for i, field in enumerate(HATEBASE_FIELDS):
    print field
    train_and_eval_auc( train_x, train_y.iloc[:,i], test_x, test_y.iloc[:,i], SoftmaxClassifier() )



