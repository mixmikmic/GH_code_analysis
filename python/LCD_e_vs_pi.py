import os
import random
gpu = 'gpu0'
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
from sklearn.metrics import roc_curve, roc_auc_score
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
get_ipython().magic('matplotlib inline')

class Classification_Generator:
    '''
    Data generator class for directory of h5 files
    '''

    def __init__( self, batch_size,train_split=0.7,validation_split=0.2,test_split=0.1):
        self.batch_size = batch_size
        self.filelist=[]
        for i in xrange(1,161):
            self.filelist.append('/data/kaustuv1993/NewLCD/ChPiEle_Shuffled/ChPiEle_%d.h5'%i) 
        self.train_split = train_split 
        self.validation_split = validation_split
        self.test_split = test_split
        self.fileindex = 0
        self.filesize = 0
        self.position = 0
    #function to call when generating data for training

  
    def train(self,modeltype=3):
        '''
        Generate data for training only
        '''
        length = len(self.filelist)
        #deleting the validation and test set filenames from the filelist
        del self.filelist[np.floor((1-(self.train_split))*length).astype(int):]
        return self.batches(modeltype)
    #function to call when generating data for testing


    def test(self,modeltype=3):
        '''
        Generate data for testing only
        '''
        length = len(self.filelist)
        #deleting the train and validation set filenames from the filelist
        del self.filelist[:np.floor((1-self.test_split)*length).astype(int)+1]
        return self.batches(modeltype)
    #function to call when generating data for validating


    def validation(self,modeltype=3):
        '''
        Generate data for validation only
        '''
        length = len(self.filelist)
        #modifying the filename list to only include files for validation set
        self.filelist = self.filelist[np.floor(self.train_split*length+1).astype(int):np.floor((self.train_split+self.validation_split)*length+1).astype(int)]
        return self.batches(modeltype)


        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Reads files to gather data until batch size is satisfied, then yeilds
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        self.filesize = np.array(f['ECAL']).shape[0]


        if (position + batch_size < self.filesize):
            data_ECAL = np.array(f['ECAL'][position : position + batch_size])
            #data_HCAL = np.array(f['HCAL'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size][:,0])
            position += batch_size
            f.close()
            return data_ECAL, target, fileindex, position
        
        else:

            data_ECAL = np.array(f['ECAL'][position : ])
            #data_HCAL = np.array(f['HCAL'][position : ])
            target = np.array(f['target'][position:][:,0])
            #target = np.delete(target,0,1)
            f.close()
            

            if (fileindex+1 < len(self.filelist)):
                if(self.batch_size-data_ECAL.shape[0]>0):
                    while(self.batch_size-data_ECAL.shape[0]>0):
                        if(int(np.floor((self.batch_size-data_ECAL.shape[0])/self.filesize))==0):
                            number_of_files=1
                        else:
                            number_of_files=int(np.ceil((self.batch_size-data_ECAL.shape[0])/self.filesize))
                        for i in xrange(0,number_of_files):

                            if fileindex + i + 1 > len(self.filelist):
                                fileindex = -1 - i

                            f = h5py.File(self.filelist[fileindex+i+1],'r')

                            if (self.batch_size-data_ECAL.shape[0]<self.filesize):
                                position = self.batch_size-data_ECAL.shape[0]
                                data_temp_ECAL = np.array(f['ECAL'][ : position])
                                #data_temp_HCAL = np.array(f['HCAL'][: position])
                                target_temp = np.array(f['target'][:position][:,0])
#target_temp = np.array(f['target'][:position][:,1]) for regression
                            else:
                                data_temp_ECAL = np.array(f['ECAL'])
                                #data_temp_HCAL = np.array(f['HCAL'])
                                target_temp = np.array(f['target'][:,0])

                            f.close()
                            data_ECAL = np.concatenate((data_ECAL, data_temp_ECAL), axis=0)
                            #data_HCAL = np.concatenate((data_HCAL, data_temp_HCAL), axis=0)
                            target = np.concatenate((target, target_temp), axis=0)

                    if (fileindex +i+1<len(self.filelist)):
                        fileindex = fileindex +i+1
                    else:
                        fileindex = 0
                else:
                    position = 0
                    fileindex=fileindex+1
            else:
                fileindex = 0
                position = 0
            
            return data_ECAL, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size

    def batches(self, modeltype):
        '''
        Loops indefinitely and continues to return data of specified batch size
        '''
        while (self.fileindex < len(self.filelist)):
            data_ECAL, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data_ECAL.shape[0]!=self.batch_size:
                continue

            if modeltype==3:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(1, 25, 25, 25))
                #data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(1, 5, 5, 60))

            elif modeltype==2:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(25, 25, 25))
                data_ECAL = np.swapaxes(data_ECAL, 1, 3)
                #data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(4, 4, 60))
                #data_HCAL = np.swapaxes(data_HCAL, 1, 3)

            elif modeltype==1:
                data_ECAL= np.reshape(data_ECAL,(self.batch_size,-1))
                #data_HCAL= np.reshape(data_HCAL,(self.batch_size,-1))

            yield (data_ECAL, target)
        self.fileindex = 0[2]

def loadmodel(name, weights = False):
    json_file = open('%s.json'%name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model
    if weights==True:
        model.load_weights('%s.h5'%name)
    #print (model.summary())
    print("Loaded model %s from disk"%name)
    return model

def savemodel(model,name="neural network"):

    model_name = name
    #model.summary()
    model.save_weights('/data/kaustuv1993/NewLCD/Models/Regression/%s_w.h5'%model_name, overwrite=True)
    model_json = model.to_json()
    with open("/data/kaustuv1993/NewLCD/Models/Regression/%s_m.json"%model_name, "w") as json_file:
        json_file.write(model_json)
        
def savelosses(hist, name="neural network"):    
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])
    f = h5py.File("/data/kaustuv1993/NewLCD/Models/Regression/%s_h.h5"%name,"w")
    f.create_dataset('loss',data=loss)
    f.create_dataset('val_loss',data=valoss)
    f.close()

#scan over various models (with full z,y,x depths respectively) with different fractions of training set
ts = [0.01,0.1,0.25,0.5,0.7]
tsn = ["0pt01", "0pt1", "0pt25", "0pt5", "0pt7"]
for j in xrange(1,6):
    dim1 = [5, 5, 25]
    dim2 = [5, 25, 5]
    dim3 = [25, 5, 5]
    scan = ["xyscan", "xzscan","yzscan"]
    pool = [(2,2,1),(2,1,2),(1,2,2)]
    for i in xrange(0,3):
        model = Sequential()
        model.add(Convolution3D(25, dim1[i], dim2[i], dim3[i], input_shape = (1, 25, 25, 25), activation='relu'))
        model.add(MaxPooling3D(pool_size=pool[i]))
        model.add(Flatten())
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd')
        model.summary()
        f=open("/data/kaustuv1993/NewLCD/Models/Classification/Classification_Training_Log.txt","w")
        
        ds = Classification_Generator(400, train_split=ts[j])
        vs = Classification_Generator(400)
        modelname="cnn3D_%s_cls_ChPiEle_ts%s"%(scan[i],tsn[j])
        check = ModelCheckpoint(filepath="/data/kaustuv1993/NewLCD/Models/Classification/%s_check.h5"%modelname, verbose=1)
        early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        hist = model.fit_generator(ds.train(modeltype=3), samples_per_epoch=50000, nb_epoch=1000, validation_data= vs.validation(modeltype=3), nb_val_samples=50000, verbose=1, callbacks=[check,early])
        savelosses(hist,name=modelname)
        f.write("%s has trained, and early-stopped"%modelname)
        savemodel(model,name=modelname)

