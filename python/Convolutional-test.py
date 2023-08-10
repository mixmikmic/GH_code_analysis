#Import all necessary packages
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
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge

#ConvNet - 3D, 6/7
#Just taking the ECAL data (24x24x25)
cnn3d_1 = Sequential()
cnn3d_1.add(Convolution3D(10, 4, 4, 5, input_shape = (1, 24, 24, 25), activation='sigmoid'))
cnn3d_1.add(Convolution3D(3, 2, 2, 5, activation='sigmoid'))
cnn3d_1.add(MaxPooling3D())
cnn3d_1.add(Flatten())

#Dense layer
cnn3d_1.add(Dense(1000, activation='sigmoid'))
#cnn3d_1.add(Dropout(0.5))
cnn3d_1.add(Dense(1, activation='sigmoid'))
cnn3d_1.compile(loss='mse', optimizer='sgd')
cnn3d_1.summary()

#ConvNet - 2D, 7/7
cnn2d_1 = Sequential()
cnn2d_1.add(Convolution2D(10, 4, 4, input_shape = (25, 20, 20), activation='sigmoid'))
cnn2d_1.add(MaxPooling2D())
cnn2d_1.add(Flatten())

#Dense layer
cnn2d_1.add(Dense(10000, activation='sigmoid'))
cnn2d_1.add(Dropout(0.25))
cnn2d_1.add(Dense(1, activation='linear'))
cnn2d_1.compile(loss='mse', optimizer='sgd')
cnn2d_1.summary()

'''
Class definition of Generator which reads data from input file, splits it into training and validation sets and feeds
it into the neural network to train and validate it.
'''

class RegGen:
    '''
    Data generator class for directory of h5 files
    '''

    def __init__(self, batch_size ,train_split=0.6,validation_split=0.2,test_split=0.2):
        self.batch_size = batch_size
        self.filelist=[]
        for i in xrange(1,6):
            for j in xrange(1,11):
                self.filelist.append('/data/shared/LCD/New_Data_Shuffled/GammaEscan_%d_%d.h5'%(i,j)) 
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        self.fileindex = 0
        self.filesize = 0
        self.position = 0

    def train(self,modeltype=3):
        '''
        Generate data for training only
        '''
        length = len(self.filelist)
        #deleting the validation and test set filenames from the filelist
        del self.filelist[int(np.floor((1 - self.train_split) * length)):]
        return self.batches(modeltype)
    def test(self, modeltype=3):
        '''
        Generate data for testing only
        '''
        length = len(self.filelist)
        #deleting the train and validation set filenames from the filelist
        del self.filelist[:int(np.floor((1 - self.test_split) * length)) + 1]
        return self.batches(modeltype)
    def validation(self, modeltype=3):
        '''
        Generate data for validation only
        '''
        length = len(self.filelist)
        #modifying the filename list to only include files for validation set
        self.filelist = self.filelist[int(np.floor(self.train_split*length+1)):                                      int(np.floor((self.train_split + self.validation_split) * length+1))]
        return self.batches(modeltype)
        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Reads files to gather data until batch size is satisfied, then yeilds
        '''
        f = h5py.File(self.filelist[fileindex], 'r')
        self.filesize = np.array(f['ECAL']).shape[0]

        if (position + batch_size < self.filesize):
            data_ECAL = np.array(f['ECAL'][position : position + batch_size])
            data_HCAL = np.array(f['HCAL'][position : position + batch_size])
            target  = np.array(f['target'][position : position + batch_size][:,:,1])
            position += batch_size
            f.close()
            return data_ECAL, data_HCAL, target, fileindex, position
        
        else:
            data_ECAL = np.array(f['ECAL'][position:])
            data_HCAL = np.array(f['HCAL'][position:])
            target =  np.array(f['target'][position:][:,:,1])
            f.close()
            
            if fileindex+1 < len(self.filelist):
                if self.batch_size - data_ECAL.shape[0] > 0:
                    while self.batch_size - data_ECAL.shape[0] > 0:

                        if int(np.floor((self.batch_size - data_ECAL.shape[0]) / self.filesize)) == 0:
                            number_of_files = 1
                        else:
                            number_of_files = int(np.ceil((self.batch_size-data_ECAL.shape[0]) / self.filesize))

                        for i in xrange(0, number_of_files):
                            # restart in file list in case we run out of files
                            if fileindex + i + 1 > len(self.filelist):
                                fileindex = -1 - i

                            f = h5py.File(self.filelist[fileindex+i+1],'r')

                            if (self.batch_size - data_ECAL.shape[0] < self.filesize):
                                position = self.batch_size - data_ECAL.shape[0]
                                data_temp_ECAL = np.array(f['ECAL'][:position])
                                data_temp_HCAL = np.array(f['HCAL'][:position])
                                target_temp = np.array(f['target'][:position][:,:,1])

                            else:
                                data_temp_ECAL = np.array(f['ECAL'])
                                data_temp_HCAL = np.array(f['HCAL'])
                                target_temp = np.array(f['target'][:,:,1])

                            f.close()
                            data_ECAL = np.concatenate((data_ECAL, data_temp_ECAL), axis=0)
                            data_HCAL = np.concatenate((data_HCAL, data_temp_HCAL), axis=0)
                            target = np.concatenate((target, target_temp), axis=0)
                    
                    if (fileindex + i + 1 < len(self.filelist)):
                        fileindex += i + 1
                    else:
                        fileindex = 0
                else:
                    position = 0
                    fileindex += 1
            else:
                fileindex = 0
                position = 0
            
            return data_ECAL, data_HCAL, target, fileindex, position
    
    def batches(self, modeltype):
        '''
        Loops indefinitely and continues to return data of specified batch size
        '''
        while (self.fileindex < len(self.filelist)):
            data_ECAL,data_HCAL, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data_ECAL.shape[0]!=self.batch_size:
                continue
            if modeltype==3:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(1, 24, 24, 25))
                data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(1, 4, 4, 60))

            elif modeltype==2:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(24, 24, 25))
                data_ECAL = np.swapaxes(data_ECAL, 1, 3)
                data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(4, 4, 60))
                data_HCAL = np.swapaxes(data_HCAL, 1, 3)

            elif modeltype==1:
                data_ECAL= np.reshape(data_ECAL,(self.batch_size,-1))
                data_HCAL= np.reshape(data_HCAL,(self.batch_size,-1))

            yield (data_ECAL,target/500.) #not returning HCAL because we are not using HCAL for the above models
        self.fileindex = 0

#Declaring objects of Data Generator to feed input from files as input dataset and validation set
ds = RegGen(1000) #input dataset
vs = RegGen(1000) #validation set
early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

#training the neural network and saving the history (training losses) in hist
hist = cnn3d_1.fit_generator(ds.train(modeltype=3), samples_per_epoch=2000, nb_epoch=5, validation_data= vs.validation(modeltype=3), nb_val_samples=2000, verbose=2)

#For prediction using the trained network
filelist=[]
#Making a filelist containing name of one of the input file

filelist.append('/data/shared/LCD/New_Data_Shuffled/GammaEscan_1_1.h5')
            
for path in filelist:
        f = h5py.File(path,'r')
        data_ECAL = np.array(f['ECAL'])
        data_HCAL = np.array(f['HCAL'])
        test_target = np.array(f['target'][:,:,1])
        f.close()
        
        data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(1, 24, 24, 25))
        data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(1, 4, 4, 60))
        
        pred = np.array(cnn3d_1.predict([data_ECAL])) #Just using ECAL
        print(pred)



