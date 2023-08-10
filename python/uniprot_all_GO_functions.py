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

new_test=pd.read_csv('..//..//bases/new_test_variants.csv')
new_test_texts = pd.read_csv('..//..//bases/new_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
new_test_final=pd.merge(new_test,new_test_texts,how="left",on="ID")

leaks=pd.read_csv('..//..//bases/s1_add_train.csv')
leaks_1=pd.DataFrame([leaks["ID"],leaks.drop("ID",axis=1).idxmax(axis=1).map(lambda x: x.lstrip('class'))])
leaks_2=leaks_1.T
leaks_2.columns=["ID","Class"]

train = pd.read_csv('..//..//bases/training_variants')
test = pd.read_csv('..//..//bases/test_variants')

train_texts = pd.read_csv('..//..//bases/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
test_texts = pd.read_csv('..//..//bases/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")

train = pd.merge(train, train_texts, how='left', on='ID')
test = pd.merge(test, test_texts, how='left', on='ID')

leaks_3=pd.merge(leaks_2,test[test.ID.isin(leaks_2.ID)])
leaks_final=pd.merge(leaks_3,test_texts[test_texts.ID.isin(leaks_3.ID)])

train_all = pd.concat([train,leaks_final]) #adding first stage
train_all

merge_match = new_test.merge(train_all, left_on=['Gene', 'Variation'], right_on = ['Gene', 'Variation'])
Index_leak = merge_match.ID_x - 1
new_test_index = [item for item in new_test_final.index if item not in list(Index_leak)]
test_no_leaks = new_test_final.iloc[new_test_index]
test_no_leaks

train_all['Substitutions_var'] = train_all.Variation.apply(lambda x: bool(re.search('^[A-Z]\\d+[A-Z*]$', x))*1)
new_train = train_all[train_all['Substitutions_var']==1]

data_all = pd.concat((new_train, test_no_leaks), axis=0, ignore_index=True)
data_all = data_all[['Class', 'Gene', 'ID', 'Variation', 'Text']] # just reordering
data_all_backup = data_all[:] ## We keep backup in case we need to use again
data_all

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

df['Gene ontology (GO)'][0]

df_new = df[df['Gene ontology (GO)'].notnull()] # don't consider genes with no molecular function



GO_terms_dict = dict(zip(df_new['Entry name'], df_new['Gene ontology (GO)']))

GO_terms_dict

# Find most common GO terms to use as features
def flatten(l): # taken from https://stackoverflow.com/questions/33900770/most-frequent-values-in-a-dictionary
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str): #replaced basestring with str for Python3
            for sub in flatten(el):
                yield sub
        else:
            yield el

All_GO_terms = list(set(flatten(GO_terms_dict.values())))
len(All_GO_terms)

features = pd.read_csv('all_GO_terms.csv')

features = features.values

features = [item for sublist in features for item in sublist]

len(features)

# initialize data with the features 
for feature in features:
    data_all[feature] = 0

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
data_all.to_csv("ALL_GO_FUNCTIONS.csv",index=False)

# Do an SVD on the molecular functions to get a reduction to 25 features
svd = TruncatedSVD(n_components=25, n_iter=20, random_state=18)
feature_columns = data_all.iloc[:,4:] #starting from the 4th column we have our features
truncated_molecular = pd.DataFrame(svd.fit_transform(feature_columns.values))

# add truncated molecular functions to our data 
data_new = pd.concat((train, test), axis=0, ignore_index=True)
data_SVD = pd.concat((data_new, truncated_molecular), axis = 1)
data_SVD

print(svd.explained_variance_ratio_.sum())

new_names = [] 
for i in range(25):
    new_names.append('molecular_SVD_'+str(i+1))

data_SVD.columns = data_SVD.columns[:4].tolist() + new_names

# Save the 25 svd's features into one file 
data_SVD.to_csv("molecular_bases/svd25_molecular_functions.csv",index=False)



