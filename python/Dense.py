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
get_ipython().magic('matplotlib inline')

dir='/home/kaustuv1993/Notebooks/models/'
for file in os.listdir(dir):
    if file.startswith('dense'):
        print file
        json_file = open('%s%s' % (dir,file), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print (model.summary())
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

dnn = Sequential()
dnn.add(Dense(10000,input_shape=(10000,), activation='sigmoid'))
dnn.add(Dense(100, activation='sigmoid'))
#dnn.add(Dropout(0.5))
dnn.add(Dense(10, activation='sigmoid'))
dnn.add(Dense(1, activation='linear'))
#sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
dnn.compile(loss='mse', optimizer='sgd')
#simple.load_weights('first_try.h5')
dnn.summary()

class My_Gen_Reg:
    #Data generator for regression over energy 
    def __init__( self, batch_size, filesize, filepattern='/data/shared/LCD/EnergyScan_Gamma_Shuffled/GammaEscan_*GeV_fulldataset.h5'):
        self.batch_size = batch_size
        self.filelist=[]
        for i in xrange(1,11):
            self.filelist.append('/data/shared/LCD/GammaEscan_shuffled_datasets/GammaEscan_%d_shuffled.h5'%i)
        
        self.train_split = 0.6 
        self.test_split = 0.2 
        self.validation_split = 0.2
        self.fileindex = 0
        self.filesize = filesize
        self.position = 0
    #function to call when generating data for training  
    def train(self, cnn=False):
        return self.batches(cnn)
    #function to call when generating data for validation 
    def validation(self, cnn=False):
        return self.batches(cnn)
    #function to call when generating data for testing  
    def test(self, cnn=False):
        return self.batches(cnn)
        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Yields batches of data of size N
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        print(self.filelist[fileindex],'first')
        if (position + batch_size < self.filesize):
            data = np.array(f['images'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size])
            target = np.delete(target,0,1)

            position += batch_size
            f.close()
            print('first position',position)
            return data, target, fileindex, position
        
        else:
            data = np.array(f['images'][position:])
            target = np.array(f['target'][position:])
            target = np.delete(target,0,1)
            f.close()
            
            if (fileindex+1 < len(self.filelist)):
                if(self.batch_size-data.shape[0]>0):
                    while(self.batch_size-data.shape[0]>0):
                        if(int(np.floor((self.batch_size-data.shape[0])/self.filesize))==0):
                            number_of_files=1
                        else:
                            number_of_files=int(np.ceil((self.batch_size-data.shape[0])/self.filesize))
                        for i in xrange(0,number_of_files):
                            if(fileindex+i+2>len(self.filelist)):
                                fileindex=0
                                number_of_files=number_of_files-i
                                i=0
                            f = h5py.File(self.filelist[fileindex+i+1],'r')
                            print(self.filelist[fileindex+i+1],'second')
                            if (self.batch_size-data.shape[0]<self.filesize):
                                position = self.batch_size-data.shape[0]
                                data_ = np.array(f['images'][ : position])
                                target_ = np.array(f['target'][:position])
                                target_ = np.delete(target_,0,1)
                            else:
                                data_ = np.array(f['images'][:])
                                target_ = np.array(f['target'])
                                target_ = np.delete(target_,0,1)
                            f.close()
                    #data_, target_, fileindex, position = self.batch_helper(fileindex + 1, 0, batch_size - self.filesize+position)
                            print( data.shape,data_.shape)
                            print( target.shape,target_.shape)
                            data = np.concatenate((data, data_), axis=0)
                            target = np.concatenate((target, target_), axis=0)
                    fileindex = fileindex +i+2
                else:
                    position = 0
                    fileindex=fileindex+1
            else:
                fileindex = 0
                position = 0
            
            return data, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size
    def batches(self, cnn):
        while (self.fileindex < len(self.filelist)):
            data, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data.shape[0]!=self.batch_size:
                continue
            if cnn==True:
                data = np.swapaxes(data, 1, 3)
                #data = np.swapaxes(data, 1, 2)
                #data = np.swapaxes(data, 0, 1)
                #data=data.reshape((data.shape[0],1,20,20,25))
                
            else:
                data= np.reshape(data,(self.batch_size,-1))
            yield (data, target/110.)
        self.fileindex = 0
            

ds = My_Gen_Reg(25000, 10000)
hist = dnn.fit_generator(ds.train(cnn=False), samples_per_epoch=75000, nb_epoch=10, verbose=1)

#vs = My_Gen_E(20000, 10000)
#early = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
hist = dnn.fit_generator(ds.train(cnn=False), samples_per_epoch=80000, nb_epoch=10, verbose=1)

savemodel(dnn,"dnn_fixed")
get_ipython().magic('matplotlib inline')
show_losses([("mse",hist)],"dnn_fixed")

json_file = open('dnn_fixed.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
dnn = model_from_json(loaded_model_json)
dnn.compile(loss='mse', optimizer='sgd')
# load weights into new model
dnn.load_weights("dnn_fixed.h5")
print("Loaded model from disk")

import math
md = []
rmsd = []
rel_error = []
for i in xrange(10,110):
    print (i)
    if i==13:
         continue
    fn =('/data/kaustuv1993/EnergyScan_Gamma/GammaEscan_%dGeV_fulldataset.h5'%i)
    f = h5py.File(fn,'r')
    test_data = np.array(f['images'])
    test_target=np.array(f['target'])
    #test_data = np.swapaxes(test_data,1,3)
    test_data= np.reshape(test_data,(10000,-1))
    test_target = np.delete(test_target,0,1)
    pred = dnn.predict(test_data)
    mean = np.mean(pred*110- test_target)
    diff = (np.mean((pred*110-test_target)**2))
    rmsde = math.sqrt((diff))
    re = np.mean(np.absolute((pred*110- test_target))/(test_target))
    print (mean, rmsde, re)
    md.append(mean)
    rmsd.append(rmsde)
    rel_error.append(re)

plt.plot(md)
plt.savefig('DNN Mean error per energy.pdf')
plt.show()

plt.plot(rmsd)
plt.savefig('DNN RMSD error per energy.pdf')
plt.show() 

plt.plot(rel_error)
plt.savefig('DNN Relative error per energy.pdf')
plt.show()

2+2

dnn2 = Sequential()
dnn2.add(Dense(10000,input_shape=(10000,), activation='sigmoid'))
dnn2.add(Dense(100, activation='sigmoid'))
dnn2.add(Dropout(0.5))
dnn2.add(Dense(10, activation='sigmoid'))
dnn2.add(Dense(1, activation='linear'))
#sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
dnn2.compile(loss='mse', optimizer='sgd')
#simple.load_weights('first_try.h5')
dnn2.summary()


def show_losses( histories,fname ):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)


    plt.legend()
    plt.yscale('log')
    plt.savefig('%s.pdf'%fname)
    plt.show()
    if not do_acc: 
	return

    #plt.figure(figsize=(10,10))
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #for i,(label,loss) in enumerate(histories):
    #    color = colors[i]
    #    if 'acc' in loss.history:
    #        plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
    #    if 'val_acc' in loss.history:
    #        plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
    #plt.legend(loc='lower right')
    #plt.savefig('%s.png'%fname)
   
    #plt.show()

import numpy as np
for i in xrange(10,110):
    if i==13:
        continue
    fn ='/data/shared/LCD/EnergyScan_Gamma_Shuffled/GammaEscan_%dGeV_fulldataset.h5'%i
    f = h5py.File(fn,'r')
    train_data = np.array(f['images'])
    train_target = np.array(f['target'])[:,1]
    #train_data = np.swapaxes(train_data, 1, 3)
    train_data= np.reshape(train_data,(10000,-1))
    print(i)
    my_fit = dnn2.fit(train_data, train_target/110., nb_epoch=10, validation_split=0.2, batch_size=1000, verbose=1)
    show_losses( [("mse",my_fit)],"dnn2_%d"%i)
    f.close()
    fname = "dnn2_file%d"%i
    loss = np.array(my_fit.history['loss'])
    valoss = np.array(my_fit.history['val_loss'])
    f = h5py.File("%s_losses.h5"%fname,"w")
    f.create_dataset('loss',data=loss)
    f.create_dataset('val_loss',data=valoss)
    f.close()

import math
md = []
rmsd = []
rel_error = []
for i in xrange(10,110):
    print (i)
    if i==13:
         continue
    fn =('/data/kaustuv1993/EnergyScan_Gamma/GammaEscan_%dGeV_fulldataset.h5'%i)
    f = h5py.File(fn,'r')
    test_data = np.array(f['images'])
    test_target=np.array(f['target'])
    #test_data = np.swapaxes(test_data,1,3)
    test_data= np.reshape(test_data,(10000,-1))
    test_target = np.delete(test_target,0,1)
    pred = dnn.predict(test_data)
    mean = np.mean(pred*110- test_target)
    diff = (np.mean((pred*110-test_target)**2))
    rmsde = math.sqrt((diff))
    re = np.mean(np.absolute((pred*110- test_target))/(test_target))
    print (mean, rmsde, re)
    md.append(mean)
    rmsd.append(rmsde)
    rel_error.append(re)
    f.close()

plt.plot(md)
plt.savefig('DNN2 Mean error per energy.pdf')
plt.show()

plt.plot(rmsd)
plt.savefig('DNN2 RMSD error per energy.pdf')
plt.show() 

plt.plot(rel_error)
plt.savefig('DNN2 Relative error per energy.pdf')
plt.show()

savemodel()

a = np.arange(1,82)
a = a.reshape(3,3,3,3)

a

a.reshape(3,-1)

for j in xrange(3):
    print ('bla')
print (j)



