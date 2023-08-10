import sys

sys.path.append('../python')

from timit import *

train_corp=prepare_corp_dir('../data/TIMIT_train.list','../TIMIT/train')
dev_corp=prepare_corp_dir('../data/TIMIT_dev.list','../TIMIT/test')
test_corp=prepare_corp_dir('../data/TIMIT_test.list','../TIMIT/core_test')

train_mlf=load_mlf('../data/mlf/TIMIT.train.align_cistate.mlf.cntk')
dev_mlf=load_mlf('../data/mlf/TIMIT.dev.align_cistate.mlf.cntk')
test_mlf=load_mlf('../data/mlf/TIMIT.core.align_cistate.mlf.cntk')

train_corp=prepare_corp(train_mlf,'../data/mlf/TIMIT.statelist','../TIMIT/train')
dev_corp=prepare_corp(dev_mlf,'../data/mlf/TIMIT.statelist','../TIMIT/test')
test_corp=prepare_corp(test_mlf,'../data/mlf/TIMIT.statelist','../TIMIT/core_test')

print 'Train utterance num: {}'.format(len(train_corp))
print 'Dev utterance num: {}'.format(len(dev_corp))
print 'Test utterance num: {}'.format(len(test_corp))

print train_corp[0].name
print train_corp[0].data
print train_corp[0].data.shape
print train_corp[0].phones
print train_corp[0].ph_lens

extract_features(train_corp, '../data/TIMIT_train.hdf5')
extract_features(dev_corp, '../data/TIMIT_dev.hdf5')
extract_features(test_corp, '../data/TIMIT_test.hdf5')

normalize('../data/TIMIT_train.hdf5')
normalize('../data/TIMIT_dev.hdf5')
normalize('../data/TIMIT_test.hdf5')

