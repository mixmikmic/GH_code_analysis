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
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso

import re
from bioservices import *
import collections
get_ipython().magic('pylab inline --no-import-all')

train = pd.read_csv('..//..//../bases/new_training_variants.csv')
test = pd.read_csv('..//..//../bases/new_test_variants.csv')

# only use gene from train data -> contains the classes
all_genes = set(train.Gene)
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

gene_entry_dict = {} # here we will keep the gene_entries together with their classes
class_dict = {}
for gene in all_genes:
    gene_classes = list(train.Class[train.Gene==gene])
    keyword = 'gene:%s+AND+organism:9606' %gene #to query database, with gene and organism 9606 is Homo Sapien (human)
    entry_name_tab = u.search(keyword, frmt='tab', limit=1, columns="entry name") 
    entry_name = [s.strip() for s in entry_name_tab.splitlines()][1] # gets the entry name from uniprot i.e. second position in tab
    gene_entry_dict[gene] = entry_name
    class_dict[entry_name] = gene_classes

gene_entry_dict

gene_entries = list(gene_entry_dict.values())
len(gene_entries)

df = u.get_df(gene_entries)
df

df_new = df[df['Gene ontology (cellular component)'].notnull()] # don't consider genes with no biological process

df_new['Gene ontology (cellular component)'] = df_new['Gene ontology (cellular component)'].apply(lambda x: x.split('; ')) #split functions based on ;
df_new['Gene ontology (cellular component)']

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

# Initialization of the 2056 new features with 0's
for terms in All_GO_terms:
    train[terms] = 0

# looping through all classes and getting terms for each class
'''terms_per_class = defaultdict(list)
for entry, terms in GO_terms_dict.items():
    if entry in class_dict:
        gene_classes = class_dict[entry]
        for gene_class in gene_classes:
            terms_per_class[gene_class].extend(terms)
           
        
terms_per_class'''

# code if we want most commons
'''counter_dict = {}
for classes in terms_per_class:
    counter_dict[classes] = Counter(terms_per_class[classes]).most_common(50)'''

# adds the molecular function GO terms to each gene in train data frame
for i in train.index:
    gene = train.Gene[i]
    gene_entry = gene_entry_dict[gene]
    if gene_entry in GO_terms_dict:
        GO_terms = GO_terms_dict[gene_entry]
        train.loc[i, GO_terms] = 1

train.shape

train

# fit the input X and output Y for the feature selection
X = train[list(All_GO_terms)]
y = train['Class']
names = X.columns

# Lasso model
lasso = Lasso(alpha=.001, random_state = 3).fit(X,y)
features_lasso = names[np.nonzero(lasso.coef_)]
len(features_lasso) # 164 in total

# saving the train set together with all features from uniprot
np.save("..//cellular_bases/features_cellular_function", features_lasso)



