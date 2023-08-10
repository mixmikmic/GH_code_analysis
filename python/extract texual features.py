import pandas as pd
import numpy as np

features = pd.read_pickle('crawl_datas/features_new.pkl')
ys_posts = pd.read_pickle('crawl_datas/ys_posts.pkl')
vc_posts = pd.read_pickle('crawl_datas/vc_posts.pkl')
mn_posts = pd.read_pickle('crawl_datas/mn_posts.pkl')
nbw_posts = pd.read_pickle('crawl_datas/nbw_posts.pkl')

ys_posts['site'] = 'ys'
vc_posts['site'] = 'vc'
mn_posts['site'] = 'mn'
nbw_posts['site'] = 'nbw'

del vc_posts['desc']

posts_dfs = [ys_posts,vc_posts,mn_posts,nbw_posts]
all_posts = pd.concat(posts_dfs)

all_posts.reset_index(inplace=True)
del all_posts['index']

all_posts

def get_status(cnames):
    sum = 0
    for cname in cnames:
        cname = ' '.join(cname)
        if cname in ['junotele','netmeds','name','trideal']:
            break
        sum += int( features.ix[cname,'acquired'])
    return float(sum)/len(cnames)


all_posts['acquired'] = all_posts['cname'].apply(get_status)

all_posts

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

from random import shuffle

from sklearn.linear_model import LogisticRegression

all_posts.reset_index(inplace=True)

labeled_posts = []
def get_labelpost(row):
    labeled_posts.append(LabeledSentence(row['post'],[row['site']+'_'+str(row['index'])]))
    labeled_posts.append(LabeledSentence(row['title'],[row['site']+'_t_'+str(row['index'])]))
#all_posts.reset_index(inplace=True)
all_posts.apply(get_labelpost,axis=1)

len(labeled_posts)

import multiprocessing

cores = multiprocessing.cpu_count()

model  = Doc2Vec(dm=1, hs=1, min_count=1,
                 window=5, size=200, negative=5, 
                 sample=1e-5, workers= cores)
model.build_vocab(labeled_posts)

from contextlib import contextmanager
from timeit import default_timer
import time
import datetime
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

print(datetime.datetime.now())
for epoch in range(10):
    with elapsed_timer() as elapser:
        shuffle(labeled_posts)
        model.train(labeled_posts)
    print 'epoch {0} completed in {1}'.format(epoch+1,elapser())

model.save('all_posts.d2v')

model = Doc2Vec.load('all_posts.d2v')

model.most_similar('housing')

model.most_similar('tastykhana')

model.most_similar('freshdesk')

model.most_similar('konotor')

model.most_similar('frilp')

labels = []
cname = []

def map_cname_label(row):
    labels.append(row['site']+'_'+str(row['index']))
    cname.append(row['cname'])
    labels.append(row['site']+'_t_'+str(row['index']))
    cname.append(row['cname'])
    
    
all_posts.apply(map_cname_label,axis=1)

cname_map = pd.DataFrame({'labels':labels,'cnames':cname})

cname_map

cname_vec = {}



def get_vecagg(row):
    global cname_vec
    vec = model.docvecs[row['labels']]
    for c in row['cnames']:
        c = ' '.join(c)
        if c in cname_vec.keys():
            prev_vec = cname_vec[c]
            cname_vec[c] = vec+prev_vec
        else:
            cname_vec[c] = vec

cname_map.apply(get_vecagg,axis=1)

cname_vec.keys()

post_cname = cname_vec.keys()
post_cname

all_cname = list(features['name'])
all_cname

set(all_cname).difference(set(post_cname))

vec_df = pd.DataFrame(cname_vec).T

vec_df

all_features = pd.concat([vec_df,features],axis=1)

all_features.to_pickle('data/all_features.pkl')

all_posts.to_pickle('data/all_posts.pkl')

len(all_features)



