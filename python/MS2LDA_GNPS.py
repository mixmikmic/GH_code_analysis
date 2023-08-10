get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')


import sys
sys.path.append('/Users/simon/git/lda/code/')
sys.path.append('/Users/simon/git/lda/gnps/')

from lda import VariationalLDA

import pickle

prefix = '/Users/simon/git/lda/gnps/gnps_'
with open(prefix+'corpus.lda','r') as f:
    corpus = pickle.load(f)

with open(prefix+'metadata.lda','r') as f:
    metadata = pickle.load(f)

with open(prefix+'fragment_masses.lda','r') as f:
    fragment_masses = pickle.load(f)
with open(prefix+'fragment_names.lda','r') as f:
    fragment_names = pickle.load(f)
with open(prefix+'fragment_counts.lda','r') as f:
    fragment_counts = pickle.load(f)

with open(prefix+'loss_masses.lda','r') as f:
    loss_masses = pickle.load(f)
with open(prefix+'loss_names.lda','r') as f:
    loss_names = pickle.load(f)
with open(prefix+'loss_counts.lda','r') as f:
    loss_counts = pickle.load(f)

feat_thresh = 2
to_remove = []
for i,fragment_name in enumerate(fragment_names):
    if fragment_counts[i]<feat_thresh:
        to_remove.append(fragment_name)
for i,loss_name in enumerate(loss_names):
    if loss_counts[i]<feat_thresh:
        to_remove.append(loss_name)
print "Found {} to remove (of {})".format(len(to_remove),len(loss_names)+len(fragment_names))  

instances_removed = 0
doc_pos = 0
sub_corpus = {}
for doc in corpus:
    sub_corpus[doc] = {}
    for word in corpus[doc]:
        if not word in to_remove:
            sub_corpus[doc][word] = corpus[doc][word]
    doc_pos += 1
    if len(sub_corpus[doc]) == 0:
        del sub_corpus[doc]
    if doc_pos % 1000 == 0:
        print "Done doc {}".format(doc_pos)

with open(prefix+'sub_corpus.lda','w') as f:
    pickle.dump(sub_corpus,f,-1)

n_words = {}
total = 0
for doc in sub_corpus:
    n_words[doc] = len(sub_corpus[doc])
    total += n_words[doc]
print "Average {} unique words per document".format(1.0*total/len(sub_corpus))

from lda import VariationalLDA
gnps_lda = VariationalLDA(sub_corpus,K=500,alpha=1,eta=0.1,update_alpha=True,normalise = 100)

gnps_lda.run_vb(n_its = 100)

gnps_lda.run_vb(n_its=900,initialise=False)

from lda_plotters import VariationalLDAPlotter

vp = VariationalLDAPlotter(gnps_lda)
vp.bar_alpha()



import networkx as nx
# Extract the topics of interest
topics = []
topic_idx = []
topic_id = []
topic_degree_thresh = 10
p_thresh = 0.05
# eth = v_lda.get_expect_theta()
# for i in range(v_lda.K):
eth = gnps_lda.get_expect_theta()
print eth.shape
for i in range(gnps_lda.K):
    s = (eth[:,i]>p_thresh).sum()
    if s > topic_degree_thresh:
        topics.append("motif_{}".format(i))
        topic_idx.append(i)
        topic_id.append(i)
        
        
        
        
G = nx.Graph()
for i,t in enumerate(topics):
    s = (eth[:,topic_idx[i]]>p_thresh).sum()
    G.add_node(t,group=2,name=t,size=5*s,
               type='circle',special=False,
              in_degree=s,score=1)
print "Added {} topics".format(len(topics))


# Add the parents
parents = []
parent_dict = {}
parent_id = {}
doc_for_later = None
j = len(topics)

eth = gnps_lda.get_expect_theta()
for doc in gnps_lda.corpus:
    parent_pos = gnps_lda.doc_index[doc]
#     parent_name = "doc_{}_{}".format(doc.mz,doc.rt)
    parent_name = metadata[doc]['compound']
    for i,t in enumerate(topics):
        topic_pos = topic_idx[i]
        this_topic_id = topic_id[i]
        if eth[parent_pos,topic_pos] > p_thresh:
            if not parent_name in parents:
                parents.append(parent_name)
                parent_dict[parent_name] = doc
                parent_id[doc] = j
                j += 1
                G.add_node(parent_name,group=1,name=parent_name,
                           size=20,type='square',peakid=parent_name,
                          special=False,in_degree=0,score=0)


            G.add_edge(t,parent_name,weight=5*eth[parent_pos,topic_pos])
            
print "Added {} parents".format(len(parents))

import json
from networkx.readwrite import json_graph
d = json_graph.node_link_data(G) 
json.dump(d, open('../joegraph/gnps_graph.json','w'),indent=2)

# Write the topics to a file
b = gnps_lda.beta_matrix.copy()
print b.shape
with open('../gnps/topics.txt','w') as f:
    for topic in topic_id:
        f.write('TOPIC: {}\n'.format(topic))
        word_tup = []
        for word in gnps_lda.word_index:
            word_tup.append((word,b[topic,gnps_lda.word_index[word]]))
        word_tup = sorted(word_tup,key = lambda x: x[1],reverse=True)
        total_prob = 0.0
        pos = 0
        n_words = 0
        while total_prob < 0.9 and n_words <= 20:
            total_prob += word_tup[pos][1]
            n_words += 1
            f.write("\t{}: {}\n".format(word_tup[pos][0],word_tup[pos][1]))
            pos += 1
        f.write('\n\n')

doc = parent_dict['Lonidamine [M+H]']
precursor_mass = float(metadata[doc]['parentmass'])
title = doc + "  (" + metadata[doc]['compound'] + ")"
vp.plot_document_topic_colour(doc,precursor_mass = precursor_mass,show_losses = True,title = title)

for doc in G.neighbors('motif_335'):
    print doc
    for word in sub_corpus[parent_dict[doc]]:
        if word.startswith('loss_46.') or word.startswith('loss_18.'):
            print word,sub_corpus[parent_dict[doc]][word]
    print
    print

m1 = 46.0057116041
m2 = 46.0053890801
1e6*abs(m1-m2)/m1

from lda_plotters import VariationalLDAPlotter
vp = VariationalLDAPlotter(gnps_lda)
thresh = 0.1
topic = 276
eth = gnps_lda.get_expect_theta()
max_found = 10
n_found = 0
di = []
for doc in gnps_lda.doc_index:
    di.append((doc,gnps_lda.doc_index[doc]))

for i,e in enumerate(eth[:,topic]):
    if e > thresh:
        n_found += 1
        doc = [d for d,j in di if j == i][0]
        title = metadata[doc]['compound']
        precursor_mass = float(metadata[doc]['parentmass'])
#         vp.plot_document_topic_colour(doc,show_losses = True,
#                                       precursor_mass=precursor_mass,title=title,
#                                       xlim = [130,140])
        vp.plot_document_colour_one_topic(doc,topic,show_losses = True,
                              precursor_mass=precursor_mass,title=title,
                              xlim = None)


    if n_found > max_found:
        break
        

print gnps_lda.corpus['CCMSLIB00000001778.ms']

pos = []
for word in gnps_lda.phi_matrix['CCMSLIB00000001778.ms']:
    print word,gnps_lda.phi_matrix['CCMSLIB00000001778.ms'][word][8]
    pos.append(gnps_lda.word_index[word])
    
print pos

print gnps_lda.beta_matrix[8,pos]
print gnps_lda.beta_matrix[5,pos]

import time
print time.clock()

m1 = 112.076118
m2 = 112.07531
1e6*abs((m1-m2)/m1)

m1 = 183.149078
m2 = 183.112717
1e6*abs((m1-m2)/m1)

with open('gnps_lda.lda','w') as f:
    pickle.dump(gnps_lda,f)

# make the dictionary

min_prob_to_keep_beta = 1e-3
min_prob_to_keep_phi = 1e-2
min_prob_to_keep_theta = 1e-2

lda_dict = {}
lda_dict['corpus'] = gnps_lda.corpus
lda_dict['word_index'] = gnps_lda.word_index
lda_dict['doc_index'] = gnps_lda.doc_index
lda_dict['K'] = gnps_lda.K
lda_dict['alpha'] = list(gnps_lda.alpha)
lda_dict['beta'] = {}
lda_dict['doc_metadata'] = metadata
wi = []
for i in gnps_lda.word_index:
    wi.append((i,gnps_lda.word_index[i]))
wi = sorted(wi,key = lambda x: x[1])

di = []
for i in gnps_lda.doc_index:
    di.append((i,gnps_lda.doc_index[i]))
di = sorted(di,key=lambda x: x[1])

ri,i = zip(*wi)
ri = list(ri)
di,i = zip(*di)
di = list(di)

    

for k in range(gnps_lda.K):
    pos = np.where(gnps_lda.beta_matrix[k,:]>min_prob_to_keep_beta)[0]
    motif_name = 'motif_{}'.format(k)
    lda_dict['beta'][motif_name] = {}
    for p in pos:
        word_name = ri[p]
        lda_dict['beta'][motif_name][word_name] = gnps_lda.beta_matrix[k,p]


eth = gnps_lda.get_expect_theta()
lda_dict['theta'] = {}
for i,t in enumerate(eth):
    doc = di[i]
    lda_dict['theta'][doc] = {}
    pos = np.where(t > min_prob_to_keep_theta)[0]
    for p in pos:
        motif_name = 'motif_{}'.format(p)
        lda_dict['theta'][doc][motif_name] = t[p]
    

# lda_dict['gamma'] = []
# for d in range(len(gnps_lda.corpus)):
#     lda_dict['gamma'].append(list(gnps_lda.gamma_matrix[d,:]))
lda_dict['phi'] = {}
ndocs = 0
for doc in gnps_lda.corpus:
    ndocs += 1
    lda_dict['phi'][doc] = {}
    for word in gnps_lda.corpus[doc]:
        lda_dict['phi'][doc][word] = {}
        pos = np.where(gnps_lda.phi_matrix[doc][word] >= min_prob_to_keep_phi)[0]
        for p in pos:
            lda_dict['phi'][doc][word]['motif_{}'.format(p)] = gnps_lda.phi_matrix[doc][word][p]
    if ndocs % 500 == 0:
        print "Done {}".format(ndocs)
        

sys.getsizeof(lda_dict)

import pickle
with open('gnps_lda.dict','w') as f:
    pickle.dump(lda_dict,f,-1)

from lda_plotters import VariationalLDAPlotter_dict
vd = VariationalLDAPlotter_dict(lda_dict)
vd.bar_alpha()

doc = lda_dict['corpus'].keys()[2]
parentmass = float(lda_dict['doc_metadata'][doc]['parentmass'])
print parentmass
vd.plot_document_colour_one_topic(doc,'motif_157',precursor_mass = parentmass)

vd.plot_document_topic_colour(doc)
print lda_dict['theta'][doc]

500*5000

import numpy as np
a = np.array([1,2,3])
type(a)
a = list(a)
type(a)
print a

from lda_plotters import VariationalLDAPlotter_dict
vd = VariationalLDAPlotter_dict(lda_dict)
G = vd.make_graph_object(filename = '../joegraph/gnps2.json')



