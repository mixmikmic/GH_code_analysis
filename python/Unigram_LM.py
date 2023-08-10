import pandas as pd
import numpy as np
import pickle
#We need this line to find the collection_vocabulary.py here, else we cannot load the col.pkl object
import sys
sys.path.append('../0_Collection_and_Inverted_Index/')
with open('../0_Collection_and_Inverted_Index/pickle/col.pkl', 'rb') as input:
    col = pickle.load(input)
inverted_index = pd.read_pickle('../0_Collection_and_Inverted_Index/pickle/inverted_index.pkl')

global_LM=inverted_index.sum(axis=1)/col.collection_length # equal: inverted_index.sum(axis=1)/inverted_index.sum(axis=1).sum()

local_LMs=inverted_index/inverted_index.sum()
# Sanity Check: Probabilities should sum columnwise to 1, and adding all columns should yield the collection size (3633)
local_LMs.sum().sum()

unigram_LM= (local_LMs.apply(lambda x: x+ global_LM)).apply(lambda x: x/2)# same as multiplying both by 0.5 and adding them
unigram_LM.to_pickle('pickle/unigramLM.pkl')#writing Unigram-LM to disk

# sanity check: probabilities in every doc should sum up to one and all docs should sum up tp 3633
unigram_LM.sum().sum() 

#sanity check: we don't want to have any negative values 
unigram_LM.min().min()<0

#sanity check: we don't want to have any zeros (since we are smoothing)
unigram_LM.isnull().sum(axis=1).sum()==0 # we check whether therer are no zeros > True intended

'''
omitted: operating in log-space to avoid numerical instability
global_LM_log_space=np.log(global_LM)
local_LMs_log_space=local_LMs.applymap(lambda x: np.log(x, out=np.zeros_like(inverted_index.as_matrix),where=x!=0))

 '''

