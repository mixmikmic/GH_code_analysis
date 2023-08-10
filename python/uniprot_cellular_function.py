# Gene Ontology can be found here: http://geneontology.org/page/ontology-documentation
import numpy as np
import pandas as pd
import string
import os
from collections import Counter
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

import re
from bioservices import *
import collections
get_ipython().magic('pylab inline --no-import-all')

train = pd.read_csv('..//..//bases/new_training_variants.csv')
test = pd.read_csv('..//..//bases/new_test_variants.csv')

data_all = pd.concat((train, test), axis=0, ignore_index=True)

all_genes = set(data_all.Gene)
print(len(all_genes))
print(all_genes)

u = UniProt()

u.debugLevel = "INFO"
u.timeout = 100   # some queries are long and requires much more time; default is 1000 seconds

gene_entry_dict = {}
class_dict = {}
for gene in all_genes:
    keyword = 'gene:%s+AND+organism:9606' %gene #to query database, with gene and organism 9606 is Homo Sapien (human)
    entry_name_tab = u.search(keyword, frmt='tab', limit=1, columns="entry name") 
    entry_name = [s.strip() for s in entry_name_tab.splitlines()][1] # gets the entry name = in second position in list
    gene_entry_dict[gene] = entry_name

gene_entries = list(gene_entry_dict.values())
len(gene_entries)

df = u.get_df(gene_entries) # searches in uniprot -> gets results back 
df

df_new = df[df['Gene ontology (cellular component)'].notnull()] # don't consider genes with no biological process

df_new['Gene ontology (cellular component)'] = df_new['Gene ontology (cellular component)'].apply(lambda x: x.split('; ')) #split functions based on ;

GO_terms_dict = dict(zip(df_new['Entry name'], df_new['Gene ontology (cellular component)']))

GO_terms_dict

# Find most common GO terms to use as features
def flatten(l): # taken from https://stackoverflow.com/questions/33900770/most-frequent-values-in-a-dictionary
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str): #replaced basestring with str for Python3
            for sub in flatten(el):
                yield sub
        else:
            yield el

All_GO_terms = set(list(flatten(GO_terms_dict.values())))
len(All_GO_terms)

# loading the XGboost most important 190 features
features = np.load("cellular_bases/features_cellular_function.npy")

len(features)

# initialize data with the features 
for feature in features:
    data_all[feature] = 0

data_all

# add 1 if the GO term is inside the gene_entry_dict for a particular gene
for i in data_all.index:
    gene = data_all.Gene[i]
    gene_entry = gene_entry_dict[gene]
    if gene_entry in GO_terms_dict:
        GO_terms = GO_terms_dict[gene_entry]
        features_inside = list(set(GO_terms).intersection(features))# get only features in the GO_terms that we need
        data_all.loc[i, features_inside] = 1

data_all.shape

data_all

# Save the 190 features into one csv file in case we will use it again
data_all.to_csv("cellular_bases/all_cellular_functions.csv",index=False)

# Do an SVD on the molecular functions to get a reduction to 25 features
svd = TruncatedSVD(n_components=25, n_iter=20, random_state=20)
feature_columns = data_all.iloc[:,4:] #starting from the 4th column we have our features
truncated_molecular = pd.DataFrame(svd.fit_transform(feature_columns.values))

# add truncated molecular functions to our data 
data_new = pd.concat((train, test), axis=0, ignore_index=True)
data_SVD = pd.concat((data_new, truncated_molecular), axis = 1)
data_SVD

print(svd.explained_variance_ratio_.sum())

new_names = [] 
for i in range(25):
    new_names.append('cellular_SVD_'+str(i+1))

data_SVD.columns = data_SVD.columns[:4].tolist() + new_names

# Save the 25 svd's features into one file 
data_SVD.to_csv("cellular_bases/svd25_cellular.csv",index=False)



