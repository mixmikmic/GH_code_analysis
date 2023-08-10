import numpy as np
import pandas as pd
import string
import os
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
from bioservices import *
import collections
get_ipython().magic('pylab inline --no-import-all')

train = pd.read_csv('..//bases/training_variants')
test = pd.read_csv('..//bases/test_variants')

data_all = pd.concat((train, test), axis=0, ignore_index=True)

all_genes = set(data_all.Gene)
print(len(all_genes))
print(all_genes)

u = UniProt()

res = u.search("ZAP70_HUMAN")
print(res)

u.debugLevel = "INFO"
u.timeout = 100   # some queries are long and requires much more time; default is 1000 seconds

# just an example of query
a = u.search('SLC16A1+AND+organism:9606', frmt='tab', limit=1,
               columns="entry name")

[s.strip() for s in a.splitlines()]

gene_entry_dict = {}
class_dict = {}
for gene in all_genes:
    keyword = 'gene:%s+AND+organism:9606' %gene #to query database, with gene and organism 9606 is Homo Sapien (human)
    entry_name_tab = u.search(keyword, frmt='tab', limit=1, columns="entry name") 
    entry_name = [s.strip() for s in entry_name_tab.splitlines()][1] # gets the entry name = in second position in list
    gene_entry_dict[gene] = entry_name

gene_entry_dict

gene_entries = list(gene_entry_dict.values())
len(gene_entries)

df = u.get_df(gene_entries)
df

df_new = df[df['Gene ontology (molecular function)'].notnull()] # don't consider genes with no molecular function

df_new['Gene ontology (molecular function)'] = df_new['Gene ontology (molecular function)'].apply(lambda x: x.split('; ')) #split functions based on ;

GO_terms_dict = dict(zip(df_new['Entry name'], df_new['Gene ontology (molecular function)']))

GO_terms_dict

# Find most common GO terms to use as features
def flatten(l): # taken from https://stackoverflow.com/questions/33900770/most-frequent-values-in-a-dictionary
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str): #replaced basestring with str for Python3
            for sub in flatten(el):
                yield sub
        else:
            yield el

All_GO_terms = list(flatten(GO_terms_dict.values()))
len(All_GO_terms)

GO_terms_count = collections.Counter(All_GO_terms) 
GO_most_common = GO_terms_count.most_common(30)

GO_most_common

#adding those features as dummy variables on data set
features_list = []
for common in GO_most_common:
    term = common[0]
    features_list.append(term)
    data_all[term] = 0

data_all

GO_terms_dict_filtered = {}

for entry, terms in GO_terms_dict.items():
    GO_terms_dict_filtered[entry] = list(set(terms).intersection(features_list)) #only keeps elements from feature_list

GO_terms_dict_filtered

feature_dict = {}
for gene in all_genes:
    entry = gene_entry_dict[gene]
    if entry in GO_terms_dict_filtered: 
        GO_terms = GO_terms_dict_filtered[entry]
        feature_dict[gene] = GO_terms

feature_dict

for i in data_all.index:
    print(i)
    gene = data_all.Gene[i]
    if gene in feature_dict:
        GO_terms = feature_dict[gene]
        data_all.loc[i, GO_terms] = 1

print(data_all[data_all['protein kinase binding [GO:0019901]']==1])





