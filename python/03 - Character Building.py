#import bookworm
from bookworm import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,9)

import pandas as pd
import numpy as np
import networkx as nx
import spacy

import json
import string

book = load_book('../data/raw/fellowship_of_the_ring.txt', lower=False)
sequences = get_sentence_sequences(book)

len(book.split())

remove_punctuation = lambda s: s.translate(str.maketrans('', '', string.punctuation+'’'))
words = [remove_punctuation(p) for p in book.split()]

unique_words = list(set(words))
len(unique_words)

nlp = spacy.load('en')

propernouns = [word.text for word in nlp(' '.join(unique_words)) if word.pos_ == 'PROPN']
len(propernouns)

propernouns = [p for p in propernouns if len(p) > 3]
len(propernouns)

propernouns = [p for p in propernouns if p.istitle()]
len(propernouns)

propernouns = [p for p in propernouns if not (p[-1] == 's' and p[:-1] in propernouns)]
len(propernouns)

stopwords = nltk.corpus.stopwords.words('english')

propernouns = list(set([p.title() for p in [p.lower() for p in propernouns]]) - set(stopwords))

len(propernouns)

propernouns[:10]

characters = [tuple([character + ' ']) for character in set(propernouns)]

hash_to_sequence, sequence_to_hash = get_hashes(sequences)
hash_to_character, character_to_hash = get_hashes(characters)

df = find_connections(sequences, characters)

cooccurence = calculate_cooccurence(df)

interaction_df = pd.DataFrame([[str(c1), 
                                str(c2), 
                                cooccurence[hash(c1)][hash(c2)]]
                               for c1 in characters
                               for c2 in characters],
                              columns=['source', 'target', 'value'])

interaction_df.sample(5)

G = nx.from_pandas_dataframe(interaction_df[interaction_df['value'] > 1],
                             source='source',
                             target='target')

nx.draw_networkx(G, with_labels=True)

pd.Series(nx.pagerank(G)).sort_values(ascending=False)[:10]

interaction_df = interaction_df[interaction_df['value'] > 1]

d3_dict = {'nodes': [{"id": str(id), "group": 1} for id in set(interaction_df['source'])], 
           'links': interaction_df.to_dict(orient='records')}

with open('../src/d3/bookworm.json', 'w') as fp:
    json.dump(d3_dict, fp)

get_ipython().run_cell_magic('bash', '', 'cd ../src/d3/ \npython -m http.server')

def remove_punctuation(input_string):
    '''
    Removes all punctuation from an input string

    Parameters
    ----------
    input_string : string (required)
        input string

    Returns
    -------
    clean_string : string
        clean string
    '''
    return input_string.translate(str.maketrans('', '', string.punctuation+'’'))


def extract_character_names(book):
    '''
    Automatically extracts lists of plausible character names from a book

    Parameters
    ----------
    book : string (required)
        book in string form (with original upper/lowercasing intact)

    Returns
    -------
    characters : list
        list of plasible character names
    '''
    nlp = spacy.load('en')
    stopwords = nltk.corpus.stopwords.words('english')

    words = [remove_punctuation(w) for w in book.split()]
    unique_words = list(set(words))

    characters = [word.text for word in nlp(' '.join(unique_words)) if word.pos_ == 'PROPN']
    characters = [c for c in characters if len(c) > 3]
    characters = [c for c in characters if c.istitle()]
    characters = [c for c in characters if not (c[-1] == 's' and c[:-1] in characters)]
    characters = list(set([c.title() for c in [c.lower() for c in characters]]) - set(stopwords))

    return [tuple([c + ' ']) for c in set(characters)]

book = load_book('../data/raw/fellowship_of_the_ring.txt', lower=False)

extract_character_names(book)[-10:]

get_ipython().run_cell_magic('timeit', '', 'extract_character_names(book)')

