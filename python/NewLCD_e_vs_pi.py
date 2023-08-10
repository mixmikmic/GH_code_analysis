import os
import random
gpu = 'gpu1'
file('/home/%s/.theanorc'%os.getenv('USER'),'w').write('[nvcc]\nfastmath=True\nflags =  -arch=sm_30\n[global]\n#mode=FAST_RUN\ndevice=%s\nfloatX=float32'%gpu)
import theano
import keras
print(theano.config.device)

from nn_packages import *
from io_functions import *
import numpy as np
import root_numpy as rnp
import os
import sys
import re
import glob
import h5py
import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, roc_auc_score
get_ipython().magic('matplotlib inline')

f=h5py.File("/data/kaustuv1993/NewLCD/GammaEscan_1_MERGED/GammaEscan_1_1.h5","r")
f.keys()
a=np.array(f['target'])
for i in xrange(0,100):
    print(a[i])
a.shape

count = 1 
for i in xrange(1,9):
    for j in xrange(1,11):
        f=h5py.File('/data/kaustuv1993/NewLCD/EleEscan_%d_MERGED/EleEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.array(f.get('ECAL'))
        np_HCAL = np.array(f.get('HCAL'))
        np_target = np.array(f.get('target'))
        f.close()
        print (i, j)
        with h5py.File('/data/kaustuv1993/NewLCD/EleEscan_fulldatasets/EleEscan_%d.h5'%count,'w') as hf:
            hf.create_dataset('ECAL', data=np_ECAL)
            hf.create_dataset('HCAL', data=np_HCAL)
            hf.create_dataset('target', data=np_target)
            print (count, hf['ECAL'].shape)
            hf.close()
            
        count = count+1

count = 1 

for i in xrange(1,9):
    for j in xrange(1,51,5):
        
        print ("File: ", count, i, j)
                
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.array(f.get('ECAL'))
        np_HCAL = np.array(f.get('HCAL'))
        np_target = np.array(f.get('target'))
        f.close()
        j+=1
        print (j)
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.concatenate((np_ECAL, np.array(f.get('ECAL'))), axis=0)
        np_HCAL = np.concatenate((np_HCAL, np.array(f.get('HCAL'))), axis=0)
        np_target = np.concatenate((np_target, np.array(f.get('target'))), axis=0)
        f.close()
        j+=1
        print (j)
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.concatenate((np_ECAL, np.array(f.get('ECAL'))), axis=0)
        np_HCAL = np.concatenate((np_HCAL, np.array(f.get('HCAL'))), axis=0)
        np_target = np.concatenate((np_target, np.array(f.get('target'))), axis=0)
        f.close()
        j+=1
        print (j)
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.concatenate((np_ECAL, np.array(f.get('ECAL'))), axis=0)
        np_HCAL = np.concatenate((np_HCAL, np.array(f.get('HCAL'))), axis=0)
        np_target = np.concatenate((np_target, np.array(f.get('target'))), axis=0)
        f.close()
        j+=1
        print (j)
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        np_ECAL = np.concatenate((np_ECAL, np.array(f.get('ECAL'))), axis=0)
        np_HCAL = np.concatenate((np_HCAL, np.array(f.get('HCAL'))), axis=0)
        np_target = np.concatenate((np_target, np.array(f.get('target'))), axis=0)
        f.close()
        j+=1
        print (j)
        print (i, j)
                
        with h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_fulldatasets/ChPiEscan_%d.h5'%count,'w') as hf:
            hf.create_dataset('ECAL', data=np_ECAL)
            hf.create_dataset('HCAL', data=np_HCAL)
            hf.create_dataset('target', data=np_target)
            print (count, hf['ECAL'].shape)
            hf.close()
            
        count = count+1

#concatenation before shuffling e and charged Pi
count = 1
for i in xrange(1, 81):
        
    print(i)
        
    fnEle = ('/data/kaustuv1993/NewLCD/EleEscan_fulldatasets/EleEscan_%d.h5'%i)
    fnChPi = ('/data/kaustuv1993/NewLCD/ChPiEscan_fulldatasets/ChPiEscan_%d.h5'%i)
    fEle = h5py.File(fnEle,'r')
    fChPi = h5py.File(fnChPi,'r')
        
    np_ECALChPi = np.array(fChPi.get('ECAL'))
    np_HCALChPi = np.array(fChPi.get('HCAL'))
    np_targetChPi = np.array(fChPi.get('target'))
        
    print (np_ECALChPi.shape)
        
    np_ECALEle = np.array(fEle.get('ECAL'))
    np_HCALEle = np.array(fEle.get('HCAL'))
    np_targetEle = np.array(fEle.get('target'))
        
    print (np_ECALEle.shape)
               
    fEle.close()
    fChPi.close()
        
    limChPi = int(np.floor(np_ECALChPi.shape[0]/2.0))
    print (limChPi)
        
    if (limChPi % 2 == 1):
        limChPi = (limChPi-1)
        
    limEle = int(np.floor(np_ECALEle.shape[0]/2.0))
    print (limEle)
        
    if (limEle % 2 == 1):
        limEle = (limEle-1)    
        
    np_ECALEle0 = np_ECALEle[limEle:]
    np_HCALEle0 = np_HCALEle[limEle:]
    np_targetEle0 = np_targetEle[limEle:]
        
    np_ECALChPi0 = np_ECALChPi[limChPi:]
    np_HCALChPi0 = np_HCALChPi[limChPi:]
    np_targetChPi0 = np_targetChPi[limChPi:]
        
    np_ECALEle1 = np_ECALEle[:limEle]
    np_HCALEle1 = np_HCALEle[:limEle]
    np_targetEle1 = np_targetEle[:limEle]
        
    np_ECALChPi1 = np_ECALChPi[:limChPi]
    np_HCALChPi1 = np_HCALChPi[:limChPi]
    np_targetChPi1 = np_targetChPi[:limChPi]
        
    np_ECAL0 = np.concatenate((np_ECALEle0, np_ECALChPi0), axis=0)
    np_HCAL0 = np.concatenate((np_HCALEle0, np_HCALChPi0), axis=0)
    np_target0 = np.concatenate((np_targetEle0, np_targetChPi0), axis=0)
        
    np_ECAL1 = np.concatenate((np_ECALEle1, np_ECALChPi1), axis=0)
    np_HCAL1 = np.concatenate((np_HCALEle1, np_HCALChPi1), axis=0)
    np_target1 = np.concatenate((np_targetEle1, np_targetChPi1), axis=0)
        
    with h5py.File('/data/kaustuv1993/NewLCD/EleChPi_shuffled/EleChPi_shuffled_%d.h5'%count,'w') as hf:
        hf.create_dataset('ECAL', data=np_ECAL0)
        hf.create_dataset('HCAL', data=np_HCAL0)
        hf.create_dataset('target', data=np_target0)
        print (count, hf['ECAL'].shape)
        hf.close()
        
    count = count+1
        
    with h5py.File('/data/kaustuv1993/NewLCD/EleChPi_shuffled/EleChPi_shuffled_%d.h5'%count,'w') as hf:
        hf.create_dataset('ECAL', data=np_ECAL1)
        hf.create_dataset('HCAL', data=np_HCAL1)
        hf.create_dataset('target', data=np_target1)
        print (count, hf['ECAL'].shape)
        hf.close()
            
    count = count+1

2+2

count = 0 
target_list = []
for j in xrange(1,2):
    #if i==8 and j==44:
    #    continue
    f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_fulldatasets/ChPiEscan_%d.h5' %(j), "r")
    np_ECAL = np.array(f.get('ECAL'))
    np_HCAL = np.array(f.get('HCAL'))
    np_target = np.array(f.get('target'))
    
    for i in xrange(0,np_ECAL.shape[0]):
            target_list.append(np_target[i][0])
    new_target = np.array(target_list)
    print(new_target.shape)
    #new_target = np.append(new_target,np_charge,axis=1)
    
    f.close()
    #print (np_HCAL.shape, i)
    #for k in xrange(0,np_target.shape[0]):
       #if np_target[k][0][0]==11 or np_target[k][0][0]==211 or np_target[k][0][0]==-11 or np_target[k][0][0]==1:#np_target[k][0][0]==11 or np_target[k][0][0]==-11:
            #print (i, j, k)
            #count = count+1
#print (count) 
   # print(np_target.shape)
    #print(np_target[1][0][1])

print(new_target[0])
print(np_target[0][0])

count = 0 
for i in xrange(1,9):
    for j in xrange(1,51):
        #if i==8 and j==44:
        #    continue
        f=h5py.File('/data/kaustuv1993/NewLCD/ChPiEscan_%d_MERGED/ChPiEscan_%d_%d.h5' %(i,i,j), "r")
        #np_ECAL = np.array(f.get('ECAL'))
        np_HCAL = np.array(f.get('HCAL'))
        np_target = np.array(f.get('target'))
        f.close()
        print (np_HCAL.shape, i)
        for k in xrange(0,np_target.shape[0]):
            if np_target[k][0][0]==0 or np_target[k][0][0]==211 or np_target[k][0][0]==-211 or np_target[k][0][0]==1:#np_target[k][0][0]==11 or np_target[k][0][0]==-11:
                print (i, j, k)
                count = count+1
print (count)        



