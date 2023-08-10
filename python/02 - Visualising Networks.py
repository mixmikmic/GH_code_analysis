import bookworm
from bookworm import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12,9)

import pandas as pd
import numpy as np

book = load_book('../data/raw/hp.txt')
characters = load_characters('../data/raw/characters_hp.csv')
sequences = get_character_sequences(book)

hash_to_sequence, sequence_to_hash = get_hashes(sequences)
hash_to_character, character_to_hash = get_hashes(characters)

df = find_connections(sequences, characters)
cooccurence = calculate_cooccurence(df).to_sparse()

import networkx as nx

interaction_df = pd.DataFrame([[str(c1), 
                                str(c2), 
                                cooccurence[hash(c1)][hash(c2)]]
                               for c1 in characters
                               for c2 in characters],
                              columns=['source', 'target', 'value'])

interaction_df.sample(5)

interaction_df = interaction_df[interaction_df['value'] > 2]
interaction_df.sample(5)

G = nx.from_pandas_dataframe(interaction_df,
                             source='source',
                             target='target')

nx.draw_spring(G, with_labels=True)
#plt.savefig('graph')

pd.Series(nx.pagerank(G)).sort_values(ascending=False)[:5]

a, b = nx.hits(G)
pd.Series(a).sort_values(ascending=False)[:5]

list(nx.enumerate_all_cliques(G))[-1]

comms = nx.communicability(G)

print(comms["('griphook ',)"]["('fluffy ',)"])
print(comms["('harry ', ' potter ')"]["('ron ', ' weasley ')"])

nx.dijkstra_path(G, 
                 source="('griphook ',)", 
                 target="('fluffy ',)")

d3_dict = {'nodes': [{"id": str(id), "group": 1} for id in set(interaction_df['source'])], 
           'links': interaction_df.to_dict(orient='records')}

import json

with open('../src/d3/bookworm.json', 'w') as fp:
    json.dump(d3_dict, fp)

get_ipython().run_cell_magic('bash', '', 'ls ../src/d3/')

get_ipython().run_cell_magic('bash', '', 'cat ../src/d3/index.html')

get_ipython().run_cell_magic('bash', '', 'cd ../src/d3/ \npython -m http.server')



