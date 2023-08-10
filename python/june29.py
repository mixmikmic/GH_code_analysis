from __future__ import print_function

import cntk as C
 
from   cntk.learners import sgd, learning_rate_schedule, UnitType
from   cntk.logging import ProgressPrinter        # kind of a logger
from   cntk.layers import Dense, Sequential       #preworked layers we can choose to use (or ignore)
from   cntk.io import CTFDeserializer, MinibatchSource,StreamDef,StreamDefs
from   cntk.io import INFINITELY_REPEAT 
from   cntk.train.training_session import *

import numpy as np

input_dim    = 2     #2 features to input per sample
output_dim   = 3     #3 classes/labels each sample
layers_dim   = 1     #1 hidden dimension as this is just for starting
hidden2_dim  = 5     #5 hidden nodes in the 1 hidden dimension

input_var    = C.input(input_dim,np.float32)
label_var    = C.input(output_dim,np.float32)
inputfile    = "C:\\Users\\jiwillia\\Documents\\meetup\\deepL\\CNTK-session2\\TrainData.txt"


#reader function - returns a minibatch reader to use in training and testing of a network and learner
def create_reader(pathtofile,is_training,inputsdim,outputsdim):
    return MinibatchSource(CTFDeserializer(pathtofile,StreamDefs(labels=StreamDef(field='labels',shape=outputsdim,
            is_sparse=False),
            features=StreamDef(field='features',shape=inputsdim,is_sparse=False))),randomize=is_training,
            max_sweeps=INFINITELY_REPEAT if is_training else 1)

#create the minibatch reader and mapping to the features and labels that will be mapped to file content

areader = create_reader(inputfile,True,input_dim,output_dim)

#map input containers (tensors) with the text file structure
reader_map = {
    input_var : areader.streams.features,
    label_var : areader.streams.labels
}


#let define a network and learner  - note how inputs are not included directly in the initial definition 'amodel'
#mbs, minibatch schedule, is ignored for this one as the data is so small, may be useful in other situations
amodel     = Sequential ([Dense(hidden2_dim,activation=C.sigmoid),Dense(output_dim)])
zz         = amodel(input_var) 
mbsize     = 1      #original brainscript was 1
numbatches = 18000  #same as the brainscript (500)*25 per batch
crossent   = C.cross_entropy_with_softmax(zz,label_var)
classerror = C.classification_error(zz,label_var)
learn_rate = learning_rate_schedule(0.04,UnitType.minibatch) #.04 used in brainscript
prprint    = ProgressPrinter(0)                              #0 means a geometric print schedule
mbs        = minibatch_size_schedule([1,2],3000)             #use 1 for the first 1000 samples and then 2 after that

#print("Model: ",amodel, " Inputs: ",amodel.inputs) #just to see if the network looks like our 2 in, 5 hidden etc
Trainer2   = C.Trainer(zz,(crossent,classerror), [sgd(zz.parameters,lr=learn_rate)],[prprint])

agg_loss   = 0.0    #need to add this, testing, and prediction

for ii in range(numbatches):
    mbatch = areader.next_minibatch(mbsize,input_map=reader_map)
    Trainer2.train_minibatch(mbatch)
print('training done')

#load and process test data - use different approach since CNTK offers so many ways
testinput =  "c:\\Users\\jiwillia\\Documents\\meetup\\deepL\\CNTK-session2\\testdata.txt"
#sample entry from file:    |features 1.0 1.0 |labels 1 0 0

testreader = create_reader(testinput,False,input_dim,output_dim)

test_mb_size = 1 
sample_size  = 9
mb_to_test   = sample_size
test_results = 0.0 
for tt in range(sample_size):
    mbtest = testreader.next_minibatch(test_mb_size,input_map=reader_map)
    #print(mbtest)
    eval_error = Trainer2.test_minibatch(mbtest)
    test_results += eval_error
    #print("EE : ",eval_error)
    
print((test_results/sample_size) * 100, " error percent for ",sample_size, "test samples")

#let's try to predict skipping the testing phase for now
unknown = np.array([[[9.0,1.0]]],dtype=np.float32) #equivalent to features 4.0, 7.0 labels -1,-1,-1 if this was in a file

prediction = zz.eval({input_var : unknown})

print("predicted array : ",prediction)
print("softmax of predicted is: ",np.argmax(prediction)," :as % ",C.softmax(prediction).eval())
print('done...')

