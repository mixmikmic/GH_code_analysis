import numpy as np
import pandas as pd
import string
import os
from collections import Counter
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFE, f_regression, SelectFromModel, RFECV, SelectKBest, chi2
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
from bioservices import *
import collections
get_ipython().magic('pylab inline --no-import-all')

new_test=pd.read_csv('..//..//..//bases/new_test_variants.csv')
new_test_texts = pd.read_csv('..//..//..//bases/new_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
new_test_final=pd.merge(new_test,new_test_texts,how="left",on="ID")

leaks=pd.read_csv('..//..//..//bases/s1_add_train.csv')
leaks_1=pd.DataFrame([leaks["ID"],leaks.drop("ID",axis=1).idxmax(axis=1).map(lambda x: x.lstrip('class'))])
leaks_2=leaks_1.T
leaks_2.columns=["ID","Class"]

train = pd.read_csv('..//..//..//bases/training_variants')
test = pd.read_csv('..//..//..//bases/test_variants')

train_texts = pd.read_csv('..//..//..//bases/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")
test_texts = pd.read_csv('..//..//..//bases/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"], encoding = "utf-8")

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

train = new_train

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

gene_entries = list(gene_entry_dict.values())
len(gene_entries)

df = u.get_df(gene_entries)
df

df['Gene ontology (GO)'][0]

GO_terms_dict = dict(zip(df['Entry name'], df['Gene ontology (GO)']))

GO_terms_dict

# Find most common GO terms to use as features
def flatten(l): # taken from https://stackoverflow.com/questions/33900770/most-frequent-values-in-a-dictionary
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str): #replaced basestring with str for Python3
            for sub in flatten(el):
                yield sub
        else:
            yield el


All_GO_terms = list(set((flatten(GO_terms_dict.values())))) # we want list of the unique values (set) to use for modelling
len(All_GO_terms)

# Initialization of the 3327 new features with 0's
for terms in All_GO_terms:
    train[terms] = 0

del train['Substitutions_var']
train

train.index = range(len(train))

# adds the molecular function GO terms to each gene in train data frame
for i in train.index:
    gene = train.Gene[i]
    gene_entry = gene_entry_dict[gene]
    if gene_entry in GO_terms_dict:
        GO_terms = GO_terms_dict[gene_entry]
        train.loc[i, GO_terms] = 1

train.shape

pd.DataFrame(All_GO_terms).to_csv("all_GO_terms.csv",index=False)





# fit the input X and output Y for the feature selection
X = train[All_GO_terms]
y = train['Class']
names = X.columns
ranks = {}

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

# Lasso model
lasso = Lasso(alpha=.002, random_state = 3).fit(X,y)
features_lasso = names[np.nonzero(lasso.coef_)]
features_lasso # 182 in total

# L1-SVC model
lsvc = LinearSVC(C=0.02, penalty="l1", dual=False, random_state = 3).fit(X, y)
features_lsvc = names[np.nonzero(lsvc.coef_)[1]]
features_lsvc # 209 in total

features_lsvc.intersection(features_lasso)

forest = ExtraTreesClassifier(n_estimators=200,
                              random_state=6)
forest.fit(X, y)
model = SelectFromModel(forest, prefit=True)
X_new = model.transform(X)
X_new.shape # reduced to 822 features





feature_index



train[train.ix[:,1760]==1]

# saving the train set together with all features from uniprot
train.to_csv("train_uniprot.csv",index=False)

# loading the XGboost most important 190 features
feature_scores = np.load("features_ranking.npy")

features = []
for feature_score in feature_scores:
    feature = feature_score[0]
    features.append(feature)

features

# adding only the 190 most important features from XGboost + the dummy variables of gene
train_features = train[features]
train_original = pd.read_csv('..//bases/training_variants')
train_dummy = pd.get_dummies(train_original.Gene) 
train_new = pd.concat([train_original, train_features, train_dummy], axis=1)
train_new.shape

train_new

# save train_new somewhere

