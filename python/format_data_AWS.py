import time
beginning_time = time.time()

import sys
sys.path.append("../Code/")
from utils import performance

import os
DATA_DIR = os.path.join('..', 'Data')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# give the option for practice mode (i.e. don't load the full dataset)
practice = False

# load data
if practice:
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), nrows=1000)
    valid = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'), nrows=1000)
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), nrows=1000)
else:
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    valid = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train_X = train.drop(['click','bidprice', 'payprice'], axis=1)
train_y = train[['click','bidprice', 'payprice']].copy()

valid_X = valid.drop(['click','bidprice', 'payprice'], axis=1)
valid_y = valid[['click','bidprice', 'payprice']].copy()

del train, valid

test_X = test

del test

# columns to be removed
remove = ["bidid", "userid", "IP", "url", "urlid"]

train_X = train_X.drop(remove, axis=1)
valid_X = valid_X.drop(remove, axis=1)
test_X = test_X.drop(remove, axis=1)

# fix usertag (from lists to 1 hot encode)
train_X.usertag = train_X.usertag.apply(lambda x: x.split(","))
valid_X.usertag = valid_X.usertag.apply(lambda x: x.split(","))
test_X.usertag = test_X.usertag.apply(lambda x: x.split(","))

sss= set()
for i in train_X.usertag:
    sss |= set(i)
print("there are {} usertags in total train set".format(len(sss)))

ttt= set()
for i in valid_X.usertag:
    ttt |= set(i)
print("there are {} usertags in total valid set".format(len(ttt)))

uuu= set()
for i in test_X.usertag:
    uuu |= set(i)
print("there are {} usertags in total test set".format(len(uuu)))

len(sss-ttt)

len(ttt-uuu)

# slotID and domain are too big to get dummies for 

#-> instead keep the top 100 from each

from collections import defaultdict as dd

slid = dd(int)
dom = dd(int)

for x in train_X.slotid:
    slid[x]+=1
    
for x in train_X.domain:
    dom[x]+=1
    
    
n = 5000

# there are too many "domain" and "slotid"
# we only keep the ones with frequency over 5000 in the training set

keep_slotid = set()
keep_domain = set()

for a,b in slid.items():
    if b>5000:
        keep_slotid |= {a}
        
for a,b in dom.items():
    if b>5000:
        keep_domain |= {a}

len(keep_domain)

len(keep_slotid)

def my_map(x, S):
    if x in S:
        return(x)
    else:
        return("null")

train_X.slotid = train_X.slotid.apply(lambda x: my_map(x, keep_slotid))
valid_X.slotid = valid_X.slotid.apply(lambda x: my_map(x, keep_slotid))
test_X.slotid = test_X.slotid.apply(lambda x: my_map(x, keep_slotid))

train_X.domain = train_X.domain.apply(lambda x: my_map(x, keep_domain))
valid_X.domain = valid_X.domain.apply(lambda x: my_map(x, keep_domain))
test_X.domain = test_X.domain.apply(lambda x: my_map(x, keep_domain))

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

train_X = train_X.join(pd.DataFrame(mlb.fit_transform(train_X.pop('usertag')),
                          columns=mlb.classes_,
                          index=train_X.index))

valid_X = valid_X.join(pd.DataFrame(mlb.fit_transform(valid_X.pop('usertag')),
                          columns=mlb.classes_,
                          index=valid_X.index))

test_X = test_X.join(pd.DataFrame(mlb.fit_transform(test_X.pop('usertag')),
                          columns=mlb.classes_,
                          index=test_X.index))

train_X.columns==valid_X.columns

train_X.columns==test_X.columns

len(train_X.columns)

len(valid_X.columns)

len(test_X.columns)

time.time()-beginning_time

# columns to get dummies for (the others make more sense to keep asis)
dummy = ["useragent", "region", "city", "adexchange", "domain", "slotid", "slotvisibility", "slotformat", 
        "creative", "keypage"]

#dummy = ["useragent", "region", "city", "adexchange", "slotvisibility", "slotformat"]

train_X = pd.get_dummies(train_X, columns=dummy)#, sparse=True)

len(train_X.columns)

time.time()-beginning_time

test_X = pd.get_dummies(test_X, columns=dummy)#, sparse=True)
valid_X = pd.get_dummies(valid_X, columns=dummy)#, sparse=True)

time.time()-beginning_time

sys.getsizeof(train_X)/1000000000

# make sure columns are aligned across the three sets

# need to clean up the validation set so it is consistent with the training set


#-> remove additional columns
drop_cols = [x for x in valid_X.columns if x not in train_X.columns]
valid_X = valid_X.drop(drop_cols, axis=1)


#-> fill in zeros for missing columns
missing = [x for x in train_X.columns if x not in valid_X.columns]
for it in missing:
    valid_X[it]=0


#-> update order to same
valid_X = valid_X[list(train_X.columns)]

# need to clean up the validation set so it is consistent with the training set


#-> remove additional columns
drop_cols = [x for x in test_X.columns if x not in train_X.columns]
test_X = test_X.drop(drop_cols, axis=1)


#-> fill in zeros for missing columns
missing = [x for x in train_X.columns if x not in test_X.columns]
for it in missing:
    test_X[it]=0


#-> update order to same
test_X = test_X[list(train_X.columns)]



time.time()-beginning_time

len(train_X.columns)

sum(train_X.columns==valid_X.columns)

sum(train_X.columns==test_X.columns)

# want to save the order of the columns for future reference
itemlist=train_X.columns.tolist()

itemlist

import pickle

### save files

#train_X.to_pickle(os.path.join(DATA_DIR,'train_X'))
valid_X.to_pickle(os.path.join(DATA_DIR,'valid_X'))
test_X.to_pickle(os.path.join(DATA_DIR,'test_X'))

train_y.to_pickle(os.path.join(DATA_DIR,'train_y'))
valid_y.to_pickle(os.path.join(DATA_DIR,'valid_y'))

# train_X is too big (>4GB )so we do in two attempts
n = len(train_X)//2

train_X_1 = train_X[0:n]
train_X_2 = train_X[n:]

train_X_1.to_pickle(os.path.join(DATA_DIR,'train_X_1'))
train_X_2.to_pickle(os.path.join(DATA_DIR,'train_X_2'))

len(train_X) - len(train_X_1) - len(train_X_2)

# load using

#df = pd.read_pickle(file_name)

time.time()-beginning_time



