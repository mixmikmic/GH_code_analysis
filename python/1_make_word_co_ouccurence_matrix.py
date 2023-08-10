# you need to modify this!

# where to read the opinion file from
op_dir = '/Users/iaincarmichael/data/word_embed/scotus/opinions/' 

from future.utils import iteritems

import glob
from itertools import combinations, chain
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

import string
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize

# import local code
import sys, os
sys.path.append(os.getcwd() + '/code/')
from save import save_matrix, save_vocabulary
from courtlistener import json_to_dict


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

json_files = glob.glob(op_dir + "*.json")

# select a subset of the text files to process 
# this makes things go faster -- comment out if you want to process all the text files
np.random.seed(245)
json_files = np.random.choice(json_files, size=1000)

def docs2sentences_word_coo(json_files):
    """
    Creates the word co-occurence matrix counting the number of times each word occurs in the same sentences
    
    Parameters
    ----------
    json_files: list of paths to raw opinion json files
    
    Output
    ------
    co_counts: sparse (csr_matrix) counting the number of times pairs of words co-occur in the same sentence
    
    vocab: a list of the vocabulary in giving the row/column order for co_counts
    
    word_counts: list counting the number of times each word appears in the indexed by vocab
    """

    # use to remove punctuation from text
    kill_punct = dict((ord(char), None) for char in string.punctuation)
    
    word_counter = Counter() # single words
    pair_counts = Counter() # word pairs
    

    for f in json_files:

        # get opinion text
        text = json2text(f)

        # process opinion text and tokenize sentences into words
        sentences_word_tok = text2tok_sentences(text, kill_punct)
        
        # count numer of times each word appears
        word_counter.update(chain(*sentences_word_tok))

        # Get a list of all of the combinations
        extended = [tuple(combinations(s, 2)) for s in sentences_word_tok]
        extended = chain(*extended)

        # Sort the combinations so that A,B and B,A are treated the same
        extended = [tuple(sorted(d)) for d in extended]

        # count the combinations
        pair_counts.update(extended)

    # get vocabulary
    vocab = list(zip(*pair_counts.keys()))
    vocab = list(set(vocab[0]).union(set(vocab[1]))) 
    w2i = {vocab[i]: i for i in range(len(vocab))}

    # construct counts as lil matrix but return them as csr_matrix
    co_counts = lil_matrix((len(vocab), len(vocab)))
    for p, c in pair_counts.items():
        co_counts[w2i[p[0]], w2i[p[1]]] = c

    co_counts = (co_counts + co_counts.T).tocsr()    
    
    return co_counts, vocab, [word_counter[vocab[i]] for i in range(len(vocab))]

def json2text(path):
    """
    Given a path to a json opinion file from CourtListener, returns the text of the opinion
    """

    # read json file, parse html and get the text
    html = BeautifulSoup(json_to_dict(path)['html_with_citations'], 'lxml')

    return html.get_text()   

def text2tok_sentences(text, char_map={}):
    """
    Processes and tokenizes a document
    
    - lower case words
    - tokenize into sentences
    - remove \n
    - remove punctuation
    - remove sentences fewer than 5 charaters
    - tokenize sentences into words
    
    Parameters
    ----------
    text: the document as a string
    char_map: a dict mapping characters with s.translate(char_map) e.g. 
    can be used to remove punctuation
    
    Output
    ------
    A list of lists
    Outer list is for sentences
    Inner list is for each word in the sentence
    
    """

    # lowercase text
    text = text.lower()

    # tokenize text into sentences
    sentences = sent_tokenize(text)

    # remove \n characters
    # remove sentences with fewer than 5 character
    # remove punctuation
    sentences = [s.strip('\n').translate(char_map) for s in sentences if len(s) >= 5]


    return [[w for w in word_tokenize(s)] for s in sentences]

get_ipython().run_cell_magic('time', '', '\n# takes about 30 min for 10,000 files\nco_counts, vocab, word_counts = docs2sentences_word_coo(json_files)')

# save the data!
save_matrix('data/co_counts_small_ex', co_counts)

save_vocabulary('data/vocab_small_ex.txt', vocab)

np.save('data/word_counts_small_ex', word_counts)



