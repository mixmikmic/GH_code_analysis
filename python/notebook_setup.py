import os
from collections import Counter

import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six import iteritems

import gensim
from fuzzywuzzy import fuzz, process
from Litho.nlp_funcs import *
from Litho.similarity import (check_similarity, match_lithcode, jaccard_similarity, 
                              calc_similarity_score, print_sim_compare, merge_similar_words)
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

path = 'example_data'
lith_data = 'sampled_bores.csv'
path_to_data = os.path.join(path, lith_data)

lith_df = pd.read_csv(path_to_data, index_col='HydroCode')

path = 'example_data'
lith_data = 'sampled_bores.csv'
path_to_data = os.path.join(path, lith_data)

lith_df = pd.read_csv(path_to_data, index_col='HydroCode')

lith_df.loc[:, ['MajorLithCode', 'Description']].head()

is_unknown_or_numeric = (lith_df.MajorLithCode == 'UNKN') | lith_df.MajorLithCode.str.isnumeric()
lith_df.loc[is_unknown_or_numeric, 'MajorLithCode'].count()

lith_df.Description = lith_df.Description.str.replace('clayey', 'clay')  # Manually replace 'clayey'
lith_df.Description = lith_df.Description.str.replace('caly', 'clay')  # Manually replace mispelt 'clay'
lith_df.Description = lith_df.Description.str.replace('ravel', 'gravel')  # Manually replace mispelt 'gravel'

lith_df.loc[is_unknown_or_numeric, ['BoreID', 'MajorLithCode', 'Description']].head(10)

print(len(lith_df.loc[~is_unknown_or_numeric, 'MajorLithCode'].unique()), "Unique LithCodes that are not UNKN or numeric")
print(lith_df.loc[~is_unknown_or_numeric, 'MajorLithCode'].unique())

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
stopw2 = ['redish', 'reddish', 'red', 'black', 'blackish', 'brown', 'brownish',
          'blue', 'blueish', 'orange', 'orangeish', 'gray', 'grey', 'grayish',
          'greyish', 'white', 'whiteish', 'purple', 'purpleish', 'yellow',
          'yellowish', 'green', 'greenish', 'light', 'very', 'pink','coarse',
          'fine', 'medium', 'hard', 'soft', 'coloured', 'multicoloured',
          'weathered', 'fractured', 'dark', 'color', 'colour', 'clean', 'cleaner']

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(stopw2)  # add the additional stopwords above

subset = lith_df.loc[is_unknown_or_numeric, 'Description'].values 

print("Attempting to fill in {} unknown MajorLithCode based on provided descriptions".format(len(subset)))
unkn_cache = {}
for i in tqdm(subset):
    
    if i not in unkn_cache:
        tmp_df = lith_df.loc[~is_unknown_or_numeric, ['Description', 'MajorLithCode']]
        matches = tmp_df.apply(match_lithcode, args=(i, stopw2, ), axis=1)
        counts = Counter(matches.dropna().values)
        
        try:
            lith_code, num_occur = counts.most_common()[0]
            if num_occur < 3:
                raise IndexError  # Not enough to match lith code
        except IndexError:
            continue  # Could not find any matches!
        # End try

        # print(i, f' -> {lith_code} ({num_occur})')
        unkn_cache[i] = lith_code
    else:
        lith_code = unkn_cache[i]
    # End if
    
    lith_df.at[lith_df.Description == i, 'MajorLithCode'] = lith_code  # update lith code
# End for

is_unknown_or_numeric = (lith_df.MajorLithCode == 'UNKN') | lith_df.MajorLithCode.str.isnumeric()
print("After clean up", len(lith_df.loc[~is_unknown_or_numeric, 'MajorLithCode'].unique()), "unique LithCodes that are not UNKN or numeric")
print(lith_df.loc[~is_unknown_or_numeric, 'MajorLithCode'].unique())

import warnings
totalvocab_stemmed = []
totalvocab_tokenized = []

lith_code_desc = lith_df.loc[:, ["MajorLithCode", "Description"]]
for row in lith_code_desc.itertuples():
    allwords_stemmed = tokenize_and_stem(row.Description, stopwords) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend([allwords_stemmed]) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(row.Description, stopwords)
    totalvocab_tokenized.extend([allwords_tokenized])

print(np.shape(totalvocab_tokenized),  np.shape(totalvocab_stemmed))

print("Number of entries", len(lith_code_desc.index))
lith_code_desc.head()

## used gensim instead of cosdisimilarity from sklearn due to the huge distance matrix
dictionary = gensim.corpora.Dictionary(totalvocab_stemmed)
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(once_ids)
dictionary.compactify()
print(dictionary)
# dictionary.save(path+'/dictio.dict')
# store the dictionary, for future reference

corpus = [dictionary.doc2bow(text) for text in totalvocab_stemmed]
corpus[:10]

# gensim.corpora.MmCorpus.serialize(path+'corpus.mm', corpus)  # store to disk, for later use
tf_idf = gensim.models.TfidfModel(corpus)
sims = gensim.similarities.Similarity(path, tf_idf[corpus],
                                      num_features=len(dictionary))

x, y = [], []
for n, i in enumerate(corpus[0:50]):
    dist = 1-sims[tf_idf[i]]
    # print(dist, len(dist))
    if i == 0:
        x0, y0 = 0, 0
    elif i == 1:
        x0, y0 = 0, dist[0]
    else:
        dp1p2 = dist[0] + dist[1]
        dp1pn = dp1p2 + dist[1]
        dp2pn = dist[0] + dp1p2
        A = (dp1p2**2 + dp1pn**2 - dp2pn**2)/(2*dp1p2*dp1pn)
        x0, y0 = dp1pn*np.cos(A), dp1pn*np.sin(A)
    x.append(x0)
    y.append(y0)
# End for

get_ipython().run_line_magic('matplotlib', 'inline')

# Column data to use for clustering
target_columns = ["Description"]  # "MajorLithCode", 

# Filter to unique combinations of LithCode and Description
lith_code_desc = lith_code_desc.groupby(target_columns).size().reset_index().rename(columns={0: 'count'})
warnings.warn("WARNING - Filtering to unique combinations, which may not be desirable in the future!")

lith_desc = lith_code_desc.Description
plt.figure(figsize=(10,8))
plt.scatter(x, y)
for i,n in enumerate(lith_desc[0:50]):
    plt.text(x[i],y[i],n)

plt.show()



