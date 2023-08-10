import pandas as pd
import numpy as np
import pickle
#We need this line to find the collection_vocabulary.py here, else we cannot load the col.pkl object
import sys
sys.path.append('../0_Collection_and_Inverted_Index/')
with open('../0_Collection_and_Inverted_Index/pickle/col.pkl', 'rb') as input:
    col = pickle.load(input)
inverted_index = pd.read_pickle('../0_Collection_and_Inverted_Index/pickle/inverted_index.pkl')

col.collection_size

df=(inverted_index>0).sum(axis=1) # determine N_t
raw_idf=(col.collection_size/df) # determine N/N_t
BIM= np.log10(raw_idf*0.5)
BIM.head()

# observation: in BIM 25 weights may actually become negative - we have four negative weights
sum(BIM<0)

# parameters as presented in the lecture
k=1.5
b=0.75
document_length= inverted_index.sum()
average_document_length= col.collection_length/col.collection_size # 146.20478943022295 TODO: include in project report
doc_len_div_by_avg_doc_len= document_length/average_document_length
#sanity check, should yield 3633
doc_len_div_by_avg_doc_len.sum()

weighting_bim25_nominator= inverted_index*(k+1)
weighting_bim25_nominator.shape

#the denominator is the tricky part since we have to add scalars and a vector to each column in the inverted index at the same time
weighting_bim25_denominator=inverted_index.add((doc_len_div_by_avg_doc_len*k*b), axis=1)+(k*(1-b))
weighting_bim25_denominator.shape

#merging nominator and denominator
weighting_bim25= weighting_bim25_nominator.div(weighting_bim25_denominator)
#sanity check: 29052, 3633 ?
weighting_bim25.shape

BIM25=weighting_bim25.mul(BIM, axis=0)
BIM25.to_pickle('pickle/BIM25.pkl')

# parameters as presented in paper
k=1.2
b=0.75
document_length= inverted_index.sum()
average_document_length= col.collection_length/col.collection_size # 146.20478943022295 TODO: include in project report
doc_len_div_by_avg_doc_len= document_length/average_document_length
weighting_bim25_nominator= inverted_index*(k+1)
weighting_bim25_denominator=inverted_index.add((doc_len_div_by_avg_doc_len*k*b), axis=1)+(k*(1-b))
weighting_bim25= weighting_bim25_nominator.div(weighting_bim25_denominator)
BIM25_alt=weighting_bim25.mul(BIM, axis=0)
BIM25_alt.to_pickle('pickle/BIM25_alt.pkl')

