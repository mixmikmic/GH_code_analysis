from collection_vocabulary import Collection
import pickle
col=Collection()
with open('pickle/col.pkl', 'wb') as output:
    pickle.dump(col, output)

doc_term_matrix=[]
for doc in col.collection:
    tf_vector =[]
    for word in col.vocabulary:
        n= col.collection[doc].count(word)
        tf_vector.append(n)
    doc_term_matrix.append(tf_vector)

import pandas as pd
import numpy as np
doc_term_matrix= pd.DataFrame(data=doc_term_matrix,index= col.collection.keys(),columns=col.vocabulary)
doc_term_matrix.to_pickle('pickle/doc_term_matrix.pkl')

doc_term_matrix.head(3) # this is how the doc term matrix looks like

# Sanity Check: should have dimensions 3633*29052
doc_term_matrix.shape

# some summary stats for our project report and a sanity check that would reveal any empty docs
doc_term_matrix.sum(axis=1).describe()

inverted_index= doc_term_matrix.transpose()
inverted_index.to_pickle('pickle/inverted_index.pkl') # use later for embeddings, queries, ... 

# sanity check 1
# each term should occur at least once (implied by the way we construct the index), hence min>=1
inverted_index.sum(axis=1).min()

