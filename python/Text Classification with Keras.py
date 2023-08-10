from __future__ import print_function
import numpy as np 
import tensorflow as tf 

label_num = 8

f = np.load('corpus_all_47001.npz')

train_doc = f['train_doc']

valid_doc = f['valid_doc']

test_doc = f['test_doc']



