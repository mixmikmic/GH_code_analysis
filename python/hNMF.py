from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity as cosine
from scipy.sparse.csgraph import connected_components
import itertools
import json
import numpy as np
import glob
n_topics = 10
n_top_words = 20
###fake data|
# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))
# data_samples = dataset.data

##generate data as array of strings from local .txt files
local_data = []
philes =  glob.glob("/Users/ziv/GDrive/school/math-thesis/nmf-imp/data/*.txt")
for phile in philes:
    with open(phile, 'r') as myfile:
        data=myfile.read().replace('\n', '')
        local_data.append(unicode(data, errors='ignore'))

with open('news24.json') as data_file:    
    data = json.load(data_file)

#tfdif and nmf model building
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
    
tfidf = tfidf_vectorizer.fit_transform(local_data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#nmf = NMF(n_components=n_topics).fit(tfidf)

H = nmf.components_
W = nmf.fit_transform(tfidf)
#x = WH



t = tfidf[:,:]
t.shape

import scipy
tfidf_dense = t.todense()
U, s, V = scipy.linalg.svd(tfidf_dense, full_matrices=True)

print H.shape
print W.shape

uk= U[:,0:20]
sk= s[0:20,]
vk= V[0:20,]

S = np.zeros((20, 20), dtype=complex)
S[:20, :20] = np.diag(sk)



print uk.shape
print S.shape
print vk.shape
W_svd = np.dot(uk,S)
H_svd = vk

for i in range(0,10):
    base =vk[i,:]
    name =  " ".join([tfidf_feature_names[j] for j in base.argsort()[:-n_top_words - 1:-1]])
    print name

abs(vk)

d = []
for j in base.argsort()[:-n_top_words - 1:-1]:
    print j
    d.append([tfidf_feature_names[j]])

base.argsort()[:-n_top_words - 1:-1]

S = np.zeros(tfidf_dense.shape, dtype=complex)
S[:tfidf_dense.shape[0], :tfidf_dense.shape[0]] = np.diag(s)

np.allclose(tfidf_dense, np.dot(U, np.dot(S, V)))

base =nmf.components_[20]
name =  " ".join([tfidf_feature_names[j] for j in base.argsort()[:-n_top_words - 1:-1]])
print name

dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
    
tfidf = tfidf_vectorizer.fit_transform(data_samples)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

for i in range(0,10):
    base =nmf.components_[i]
    name =  " ".join([tfidf_feature_names[j] for j in base.argsort()[:-n_top_words - 1:-1]])
    print name

import csv
with open("vocab.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(tfidf_feature_names)


weights = (5000/W.sum())*W.sum(axis=0)

def build_wgraph(alpha=2):
    if alpha != 2:
        return [[int(cosine(H[i],H[j])[0][0] > alpha) for i in range(0, len(H))] for j in range(0, len(H))]
    else:
        return [[cosine(H[i],H[j])[0][0] for i in range(0, len(H))] for j in range(0, len(H))]
def thresh_vals(numbin):
    binz = []
    w = build_wgraph(2)
    chain = itertools.chain(*w)
    s =sorted(list(chain))
    val = n_topics*n_topics/numbin
    for i,v in enumerate(s):
        if i%val ==0: binz.append(v)
    return binz

def array_distance(A,B):
    count = 0
    for i,x in enumerate(A):
        if x == B[i]:
            count+=1
    return len(A)-count

def greedy_TV_build(to_consume,bins):
    if len(to_consume)>1:
        a = to_consume[0]
        b = to_consume[1]
        ccA = connected_components(build_wgraph(a))[1]
        ccB = connected_components(build_wgraph(b))[1]
        if not np.array_equal(ccA, ccB):
            distance = array_distance(ccA,ccB)
            if distance > 8:
                new_tv = [a + i*(b-a)/bins for i in range(0,bins)]
                return new_tv + greedy_TV_build( to_consume[1:], bins)
            else:
                return [a] + greedy_TV_build(to_consume[1:], bins)
        else:
            return greedy_TV_build(to_consume[1:], bins)
    elif len(to_consume) == 1:
        return to_consume
    else:
        return []
  

def populateTree(row_level, valid_community):
    if row_level > size-2 :
        return []
    else:
        children = []
        parent_community = connected_components(build_wgraph(tv[row_level]))[1]
        child_community = connected_components(build_wgraph(tv[row_level+1]))[1]
        unique_communities = list(set(parent_community)) 
        for unique_community in unique_communities:
            if valid_community == unique_community:
                indices = [i for i, x in enumerate(parent_community) if x == unique_community] #[8,9]
                seen_communities = []
                for i in indices: #8 and 9
                    if child_community[i] in seen_communities:
                        filter(lambda x: x['community'] == str(child_community[i]), children)[0]['indices'].append(i)
                    else:
                        community_to_find = child_community[i]
                        grow_my_children = populateTree(row_level+1, community_to_find)
                        if grow_my_children:
                            name = ""
                            children.append({"community":str(child_community[i]),"indices":[i],"name" : name , "children":grow_my_children, "hasChildren": True})
                        else:
                            name = " ".join([tfidf_feature_names[j] for j in nmf.components_[i].argsort()[:-n_top_words - 1:-1]])
                            children.append({"community":str(child_community[i]),"indices":[i],"size":weights[i],"name":name, "hasChildren": False})
                        seen_communities.append(child_community[i])
        if len(children) == 1:
            try: 
                return children[0]['children']
            except:
                return []
        else:
            return children
        
def recursiveNaming(tree):
        i = tree['indices']
        base = nmf.components_[i[0]]
        if len(i) > 1:
            for ind in i[1:]:
                base = np.add(nmf.components_[ind], base)
        tree['name'] = " ".join([tfidf_feature_names[j] for j in base.argsort()[:-n_top_words - 1:-1]])
        if tree['hasChildren']:
            for child in tree['children']:
                    recursiveNaming(child)



tv = greedy_TV_build(thresh_vals(100),2)
# visualize topic tree
# for i in tv:
#     ccB = connected_components(build_wgraph(i))[1]
#     print ccB
size = len(tv)

flare = {"name" : "" , "children" : populateTree(0, 0)}
for child in flare['children']:
    recursiveNaming(child)
with open('demo.json', 'w') as outfile:
    json.dump(flare, outfile)

data

nmf.components_.shape

from dsnmf import DSNMF, appr_seminmf



