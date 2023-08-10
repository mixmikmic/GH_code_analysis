get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from termcolor import colored

face_cascade = cv2.CascadeClassifier('/home/mckc/Downloads/opencv-2.4.13/data/haarcascades_GPU/haarcascade_frontalface_default.xml')

def load_data():
    import pandas as pd
    import numpy as np
    from PIL import Image
    import cv2 
    from skimage.transform import resize
    
    train = pd.read_csv('/home/mckc/TwoClass//train.csv')
    test = pd.read_csv('/home/mckc/TwoClass//test.csv')
    print 'the training data shape is ',train.shape
    print 'the test data shape is ', test.shape
    
    train_faces = np.zeros((1,96,96),dtype=np.uint8)
    Y_train=[]
    missing = []
    multiple = []
    for i in range(train.shape[0]):
        image = np.array(cv2.imread(train.values[i,0], cv2.CV_LOAD_IMAGE_GRAYSCALE))
        #print image
        faces  = face_cascade.detectMultiScale(image,scaleFactor=1.2,minNeighbors=6,minSize=(70, 70))
        n_faces = len(faces)
        if n_faces is 1:
            for (x,y,w,h) in faces:
                fac = np.array(image)[y:(y+h),x:(x+h)]
                out = (resize(fac,(96,96))).reshape((1,96,96))
                train_faces = np.vstack((train_faces,out))
                Y_train = np.append(Y_train,train.values[i,1])
        else:
            if n_faces > 1:
                missing = np.append(missing,i)
            else:
                multiple = np.append(multiple,i)
        if i % 20==0:
            print colored((float(i)/train.shape[0]*100 ,' Percentage complete'), 'green')
    print 'missing count:',len(missing),'\nmuiltiple images count',len(multiple)
    train_faces = train_faces[1:,:,:]
    
    test_faces = np.zeros((1,96,96),dtype=np.uint8)
    Y_test = []
    file_names = []
    for i in range(test.shape[0]):
        image = np.array(cv2.imread(test.values[i,0], cv2.CV_LOAD_IMAGE_GRAYSCALE))
        faces  = face_cascade.detectMultiScale(image,scaleFactor=1.2,minNeighbors=6,minSize=(70, 70))
        n_faces = len(faces)
        if n_faces is 1:
            for (x,y,w,h) in faces:
                fac = np.array(image)[y:(y+h),x:(x+h)]
                out = (resize(fac,(96,96))).reshape((1,96,96))
                test_faces = np.vstack((test_faces,out))
                Y_test = np.append(Y_test,test.values[i,1])
                file_names = np.append(file_names,test.values[i,0])
        else:
            if n_faces > 1:
                missing = np.append(missing,i)
            else:
                multiple = np.append(multiple,i)
        if i % 20==0:
            print colored((float(i)/train.shape[0]*100 ,' Percentage complete'), 'green')
    test_faces = test_faces[1:,:,:]
    print len(missing),len(multiple)
    
    print 'the training file shape',train_faces.shape,Y_train.shape
    print 'the testing file shape',test_faces.shape,Y_test.shape
    
    return train_faces,test_faces,Y_train,Y_test,file_names

def simulate(X,Y):
    import scipy as sp
    import scipy.ndimage
    complete = np.zeros((1,96,96),dtype=np.uint8)
    Y_complete = []
    for i in range(len(X)):
        complete = np.vstack((complete,X[i,:,:].reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = 5,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = 10,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = 15,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = -5,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = -15,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(X[i,:,:], angle = -10,reshape=False,cval=1).reshape(1,96,96)))
        rotated = np.fliplr(X[i,:,:])
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = 5,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = 10,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = 15,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = -5,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = -10,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,scipy.ndimage.rotate(rotated, angle = -15,reshape=False,cval=1).reshape(1,96,96)))
        complete = np.vstack((complete,rotated.reshape(1,96,96)))
        Y_complete = np.append(Y_complete,([Y[i]]*14))
        if i % 10==0:
            print colored((float(i)/len(X)*100 ,' Percentage complete'),'green')
    complete = complete[1:,:,:]
    return complete,Y_complete

X_tr,X_tst,Y_tr,Y_tst,file_names = load_data()

import time
start_time = time.clock()
X,Y = simulate(X_tr,Y_tr)
print X.shape,Y.shape
print time.clock() - start_time, "seconds"

def standard(X):
    return (X - X.mean())/X.max()

X_test = standard(X_tst)
X = standard(X)

X_normal = X.reshape(-1,9216)
X_test_normal = X_test.reshape(-1,9216)
map, Y_number = np.unique(Y, return_inverse=True)
Y_test_numer = np.unique(Y_tst, return_inverse=True)[1]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

recognizer = RandomForestClassifier(500,verbose=0,oob_score=True,n_jobs=-1)
recognizer.fit(X_normal,Y_number)

Y_rf= recognizer.predict(X_test.reshape(-1,9216))
Y_rf_vales = map[Y_rf]

print 'Accuracy of the model is ',accuracy_score(Y_tst,Y_rf_vales)
confusion_matrix(Y_tst,Y_rf_vales)

importances = recognizer.feature_importances_
importance_image = importances.reshape(96,96)
#plt.figure(figsize=(7,7))
plt.imshow(importance_image,cmap=cm.Greys_r)

for i in range(len(Y_test_numer)):
    print file_names[i],Y_rf_vales[i]

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)


Y_Keras = np_utils.to_categorical(Y_number, 2)
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
model = Sequential()
model.add(Dense(1000, input_dim=9216,activation='sigmoid'))
#model.add(Dense(500,activation='sigmoid'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(2,activation='softmax'))
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
model.fit(X.reshape(-1,9216), Y_Keras, nb_epoch=30, batch_size=5,verbose=1
         ,validation_data=(X_test.reshape(-1,9216), np_utils.to_categorical(Y_test_numer, 2)))

Y_kr= model.predict_classes(X_test.reshape(-1,9216))
Y_kr_vales = map[Y_kr]

print 'Accuracy of the model is ',accuracy_score(Y_tst,Y_kr_vales,'\n')
confusion_matrix(Y_tst,Y_kr_vales)

import lasagne
from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
from lasagne import layers
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator,visualize,NeuralNet
#Conv2DLayer = layers.Conv2DLayer
#MaxPool2DLayer = layers.MaxPool2DLayer

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,
    hidden4_num_units=1000,
    dropout4_p=0.5,
    hidden5_num_units=1000,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,
    
    update = nesterov_momentum,
    update_learning_rate=0.001,
    update_momentum=0.9,
    max_epochs=30,
    verbose=1
)
net.fit(X.reshape(-1,1,96,96).astype(np.float32), Y_number.astype(np.uint8))

Y_las= net.predict(X_test.reshape(-1,9216))
Y_las_vales = map[Y_kr]

print 'Accuracy of the model is ',accuracy_score(Y_tst,Y_las_vales,'\n')
confusion_matrix(Y_tst,Y_las_vales)

def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    return plt
plot_loss(net)



from PIL import Image
from skimage.transform import resize
jpgfile = Image.open("/home/mckc/Downloads/1.jpg")
grey = rgb2gray(np.array(jpgfile))
faces  = face_cascade.detectMultiScale(grey.astype(np.uint8),scaleFactor=1.1,minNeighbors=3,minSize=(30, 30))
print faces


for (x,y,w,h) in faces:
    fac = np.array(grey[y:(y+h),x:(x+h)])
    out = resize(fac,(96,96))
    
plt.imshow(out,cmap=cm.Greys_r)


trial = standard(out)
print 'Linear Regression Value',map,clf.predict_proba(trial.reshape(-1,9216)),map[clf.predict((trial.reshape(-1,9216)))]
print 'Random Forest Value',map,recognizer.predict_proba(trial.reshape(-1,9216)),map[recognizer
                                                                                     .predict((trial.reshape(-1,9216)))]
print 'Lasagne Value',map,net.predict_proba(trial.reshape(-1,1,96,96).astype(np.float16)),map[net.predict((trial.reshape(-1,1,96,96).astype(np.float16)))]
print 'Keras Value',map,model.predict(trial.reshape(-1,9216).astype(np.float64))

from PIL import Image
from skimage.transform import resize
jpgfile = Image.open("/home/mckc/Downloads/2.jpg")
grey = rgb2gray(np.array(jpgfile))
faces  = face_cascade.detectMultiScale(grey.astype(np.uint8),scaleFactor=1.1,minNeighbors=4,minSize=(30, 30))
print faces


for (x,y,w,h) in faces:
    fac = np.array(grey[y:(y+h),x:(x+h)])
    out = resize(fac,(96,96))
    
plt.imshow(out,cmap=cm.Greys_r)


trial = standard(out)
print 'Linear Regression Value',map,clf.predict_proba(trial.reshape(-1,9216)),map[clf.predict((trial.reshape(-1,9216)))]
print 'Random Forest Value',map,recognizer.predict_proba(trial.reshape(-1,9216)),map[recognizer
                                                                                     .predict((trial.reshape(-1,9216)))]
print 'Lasagne Value',map,net.predict_proba(trial.reshape(-1,1,96,96).astype(np.float16)),map[net.predict((trial.reshape(-1,1,96,96).astype(np.float16)))]
print 'Keras Value',map,model.predict(trial.reshape(-1,9216).astype(np.float64))

import sys
sys.setrecursionlimit(150000)

from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')

import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(model, fid)    

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)

model = load_model('my_model.h5')



