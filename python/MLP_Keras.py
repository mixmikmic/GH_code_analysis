import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adadelta
from keras.callbacks import RemoteMonitor

import sys

sys.path.append('../python')

from data import Corpus

with Corpus('../data/mfcc_train_small.hdf5',load_normalized=True,merge_utts=True) as corp:
    train,dev=corp.split(0.9)
    
test=Corpus('../data/mfcc_test.hdf5',load_normalized=True,merge_utts=True)

tr_in,tr_out_dec=train.get()
dev_in,dev_out_dec=dev.get()
tst_in,tst_out_dec=test.get()

input_dim=tr_in.shape[1]
output_dim=np.max(tr_out_dec)+1

hidden_num=256

batch_size=256
epoch_num=100

def dec2onehot(dec):
    num=dec.shape[0]
    ret=np.zeros((num,output_dim))
    ret[range(0,num),dec]=1
    return ret

tr_out=dec2onehot(tr_out_dec)
dev_out=dec2onehot(dev_out_dec)
tst_out=dec2onehot(tst_out_dec)

print 'Samples num: {}'.format(tr_in.shape[0]+dev_in.shape[0]+tst_in.shape[0])
print '   of which: {} in train, {} in dev and {} in test'.format(tr_in.shape[0],dev_in.shape[0],tst_in.shape[0])
print 'Input size: {}'.format(input_dim)
print 'Output size (number of classes): {}'.format(output_dim)

model = Sequential()

model.add(Dense(input_dim=input_dim,output_dim=hidden_num))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=output_dim))
model.add(Activation('softmax'))

#optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
optimizer= Adadelta()
loss='categorical_crossentropy'

model.compile(loss=loss, optimizer=optimizer)

print model.summary()

from keras.utils import visualize_util
from IPython.display import SVG

SVG(visualize_util.to_graph(model,show_shape=True).create(prog='dot', format='svg'))

val=(dev_in,dev_out)

hist=model.fit(tr_in, tr_out, shuffle=True, batch_size=batch_size, nb_epoch=epoch_num, verbose=0, validation_data=val)

import matplotlib.pyplot as P
get_ipython().magic('matplotlib inline')

P.plot(hist.history['loss'])

res=model.evaluate(tst_in,tst_out,batch_size=batch_size,show_accuracy=True,verbose=0)

print 'Loss: {}'.format(res[0])
print 'Accuracy: {:%}'.format(res[1])

out = model.predict_classes(tst_in,batch_size=256,verbose=0)

confusion=np.zeros((output_dim,output_dim))
for s in range(len(out)):
    confusion[out[s],tst_out_dec[s]]+=1

#normalize by class - because some classes occur much more often than others
for c in range(output_dim):
    confusion[c,:]/=np.sum(confusion[c,:])

with open('../data/phones.list') as f:
    ph=f.read().splitlines()
    
P.figure(figsize=(15,15))
P.pcolormesh(confusion,cmap=P.cm.gray)
P.xticks(np.arange(0,output_dim)+0.5)
P.yticks(np.arange(0,output_dim)+0.5)
ax=P.axes()
ax.set_xticklabels(ph)
ax.set_yticklabels(ph)
print ''

