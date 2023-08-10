# For simple array operations
import numpy as np 

# To construct the model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Some utility for splitting data and printing the classification report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

dataset = np.loadtxt('../Data/HTRU_2.csv',delimiter=',')

print 'The dataset has %d rows and %d features' %(dataset.shape[0],dataset.shape[1]-1)

# Split into features and labels
for i in range(0,10):
    dataset = shuffle(dataset)
    
features = dataset[:,0:-1]
labels = dataset[:,-1]

traindata,testdata,trainlabels,testlabels = train_test_split(features,labels,test_size=0.3)

trainlabels = trainlabels.astype('int')
testlabels = testlabels.astype('int')

print 'Number of training samples : %d'%(traindata.shape[0])
print 'Number of test samples : %d'%(testdata.shape[0])

model = Sequential() # Our model is a simple feedforward model
model.add(Dense(64,input_shape=(8,)))  # The first layer holds the input for in which our case the there are 8 features.
model.add(Activation('relu')) # First activation layer is rectified linear unit (RELU)
model.add(Dense(256)) # Second layer has 256 neurons 
model.add(Activation('relu')) # Second RELU activation
model.add(Dense(1)) # Third layer has 1 neuron because we have only one outcome - pulsar or non pulsar
model.add(Activation('softmax')) # The Scoring layer which normalizes the scores

model.summary()

model.compile(loss='binary_crossentropy',  # Loss function for binary classification
              optimizer=SGD(), # Optimizer for learning, in this case Stochastic Gradient Descent (SGD)
             metrics=['accuracy']) # Evaluation function"

batch_size = 100
n_epochs = 10

training = model.fit(traindata,trainlabels,
                     nb_epoch=n_epochs,
                     batch_size=batch_size,
                      validation_data=(testdata, testlabels),
                     verbose=1)





