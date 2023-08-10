import sys

sys.path.append('../python')

from voxforge import *

downloadVoxforgeData('../audio')

f=loadFile('../audio/Joel-20080716-qoz.tgz')
print f.props
print f.prompts
print f.data

get_ipython().magic('xdel f')

corp=loadBySpeaker('../audio', limit=30)

addPhonemesSpk(corp,'../data/lex.tgz')

print corp.keys()

spk=corp.keys()[0]

print corp[spk]

get_ipython().magic('xdel corp')

convertCTMToAli('../data/ali.ctm.gz','../data/phones.list','../audio','../data/ali.pklz')

import gzip
import pickle
with gzip.open('../data/ali.pklz') as f:
    ali=pickle.load(f)
    
print 'Number of utterances: {}'.format(len(ali))

print ali[100].spk
print ali[100].phones
print ali[100].ph_lens
print ali[100].archive
print ali[100].audiofile
print ali[100].data

import random
from sets import Set

#make a list of speaker names
spk=set()
for utt in ali:
    spk.add(utt.spk)

print 'Number of speakers: {}'.format(len(spk))

#choose 20 random speakers
tst_spk=list(spk)
random.shuffle(tst_spk)
tst_spk=tst_spk[:20]


#save the list for reference - if anyone else wants to use our list (will be saved in the repo)
with open('../data/test_spk.list', 'w') as f:
    for spk in tst_spk:
        f.write("{}\n".format(spk))

ali_test=filter(lambda x: x.spk in tst_spk, ali)
ali_train=filter(lambda x: not x.spk in tst_spk, ali)

print 'Number of test utterances: {}'.format(len(ali_test))
print 'Number of train utterances: {}'.format(len(ali_train))

#shuffle the utterances, to make them more uniform
random.shuffle(ali_test)
random.shuffle(ali_train)

#save the data for future use
with gzip.open('../data/ali_test.pklz','wb') as f:
    pickle.dump(ali_test,f,pickle.HIGHEST_PROTOCOL)
    
with gzip.open('../data/ali_train.pklz','wb') as f:
    pickle.dump(ali_train,f,pickle.HIGHEST_PROTOCOL)    

num=int(len(ali_train)*0.05)

ali_small=ali_train[:num]

with gzip.open('../data/ali_train_small.pklz','wb') as f:
    pickle.dump(ali_small,f,pickle.HIGHEST_PROTOCOL)

corp=loadAlignedCorpus('../data/ali_train_small.pklz','../audio')

corp_test=loadAlignedCorpus('../data/ali_test.pklz','../audio')

print 'Number of utterances: {}'.format(len(corp))

print 'List of phonemes:\n{}'.format(corp[0].phones)
print 'Lengths of phonemes:\n{}'.format(corp[0].ph_lens)
print 'Audio:\n{}'.format(corp[0].data)

samp_num=0
for utt in corp:
    samp_num+=utt.data.size

print 'Length of cropus: {} hours'.format(((samp_num/16000.0)/60.0)/60.0)

import sys

sys.path.append('../PyHTK/python')

import numpy as np
from HTKFeat import MFCC_HTK
import h5py

from tqdm import *

def extract_features(corpus, savefile):
    
    mfcc=MFCC_HTK()
    h5f=h5py.File(savefile,'w')

    uid=0
    for utt in tqdm(corpus):

        feat=mfcc.get_feats(utt.data)
        delta=mfcc.get_delta(feat)
        acc=mfcc.get_delta(delta)

        feat=np.hstack((feat,delta,acc))
        utt_len=feat.shape[0]

        o=[]
        for i in range(len(utt.phones)):
            num=utt.ph_lens[i]/10
            o.extend([utt.phones[i]]*num)

        # here we fix an off-by-one error that happens very inrequently
        if utt_len-len(o)==1:
            o.append(o[-1])

        assert len(o)==utt_len

        uid+=1
        #instead of a proper name, we simply use a unique identifier: utt00001, utt00002, ..., utt99999
        g=h5f.create_group('/utt{:05d}'.format(uid))
        
        g['in']=feat
        g['out']=o
        
        h5f.flush()
    
    h5f.close()

extract_features(corp,'../data/mfcc_train_small.hdf5')
extract_features(corp_test,'../data/mfcc_test.hdf5')

def normalize(corp_file):
    
    h5f=h5py.File(corp_file)

    b=0
    for utt in tqdm(h5f):
        
        f=h5f[utt]['in']
        n=f-np.mean(f)
        n/=np.std(n)        
        h5f[utt]['norm']=n
        
        h5f.flush()
        
    h5f.close()
        

normalize('../data/mfcc_train_small.hdf5')
normalize('../data/mfcc_test.hdf5')

get_ipython().system('h5ls ../data/mfcc_test.hdf5/utt00001')

from data import Corpus
import numpy as np

train=Corpus('../data/mfcc_train_small.hdf5',load_normalized=True)
test=Corpus('../data/mfcc_test.hdf5',load_normalized=True)

g=train.get()
tr_in=np.vstack(g[0])
tr_out=np.concatenate(g[1])

print 'Training input shape: {}'.format(tr_in.shape)
print 'Training output shape: {}'.format(tr_out.shape)

g=test.get()
tst_in=np.vstack(g[0])
tst_out=np.concatenate(g[1])

print 'Test input shape: {}'.format(tst_in.shape)
print 'Test output shape: {}'.format(tst_in.shape)

train.close()
test.close()

import sklearn
print sklearn.__version__

from sklearn.linear_model import SGDClassifier

model=SGDClassifier(loss='log',n_jobs=-1,verbose=0,n_iter=100)

get_ipython().magic('time model.fit(tr_in,tr_out)')

acc=model.score(tst_in,tst_out)
print 'Accuracy: {:%}'.format(acc)

corp=loadAlignedCorpus('../data/ali_train.pklz','../audio')
extract_features(corp,'../data/mfcc_train.hdf5')
normalize('../data/mfcc_train.hdf5')



