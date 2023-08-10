import os
import cv2
import numpy as np 
import pickle

if(not(os.path.isfile("alexnet.p"))):
    print "Make sure to download the alexnet weights before running this script"

weights = pickle.load( open( "alexnet.p", "rb" ) )

tmp = weights['layer_params_states']

#Change the last layer of the AlexNet architecture weights to
#be of size 101 instead of 1000
for i in xrange(0,16):
    print tmp[i]['params']['W'].shape
    if(i==14):
        W = tmp[i]['params']['W']
        weights['layer_params_states'][i]['params']['W'] = W[0:101,:]
        weights['layer_params_states'][i]['states'][0] = weights['layer_params_states'][i]['states'][0][0:101,:]
    if(i==15):
        W = tmp[i]['params']['W']
        weights['layer_params_states'][i]['params']['W'] = W[0:101]
        weights['layer_params_states'][i]['states'][0] = weights['layer_params_states'][i]['states'][0][0:101,:]
pickle.dump( weights, open( "my_alexnet.p", "wb" ) )

weights = pickle.load( open( "my_alexnet.p", "rb" ) )

tmp = weights['layer_params_states']

#Change the last layer of the AlexNet architecture weights to
#be of size 101 instead of 1000
for i in xrange(0,16):
    #print tmp[i]['params']['W'].shape
    print tmp[i]['states'][0].shape
    print
    '''
    if(i==14):
        W = tmp[i]['params']['W']
        weights['layer_params_states'][i]['params']['W'] = W[0:101,:]
    if(i==15):
        W = tmp[i]['params']['W']
        weights['layer_params_states'][i]['params']['W'] = W[0:101]
    '''
#pickle.dump( weights, open( "my_alexnet.p", "wb" ) )

import os
import cv2
import numpy as np 
import pickle

weights = pickle.load( open( "my_alexnet.p", "rb" ) )

for i in xrange(0,16):
    print weights['layer_params_states'][i]['params']['W'].shape
    if(i==14):
        print "Final layer weight dimensions: " ,weights['layer_params_states'][i]['params']['W'].shape
    if(i==15):
        print "Final layer bias dimensions: " , weights['layer_params_states'][i]['params']['W'].shape

print weights['layer_params_states'][15]['states'][0].shape



