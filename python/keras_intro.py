get_ipython().magic('pylab inline')
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

from scipy import signal
i = np.random.randint(X_train.shape[0])



c = X_train[i,:,:]
plt.imshow(c,cmap='gray'); plt.axis('off');
plt.title('original image');
plt.figure(figsize=(18,8))
for i in range(10):
    k = -1.0 + 1.0*np.random.rand(3,3)
    c_digit = signal.convolve2d(c, k, boundary='symm', mode='same');
    plt.subplot(2,5,i+1);
    plt.imshow(c_digit,cmap='gray'); plt.axis('off');


batch_size = 128
nb_classes = 10
nb_epoch = 6

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

if K.image_dim_ordering() == 'th': #if using theano
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else: #if using tensorflow
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#sub-sample of test data to improve training speed. Comment out 
#if you want to train on full dataset.
X_train = X_train[:10000,:,:,:]
y_train = y_train[:10000]


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Create sequential convolutional multi-layer perceptron with max pooling and dropout
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
#model.add(Activation('relu')) #commented out this layer to speed up learning. 
#model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1])) #commented out this layer to speed up learning.
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#Let's see what we've constructed layer by layer
model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#choose a random data from test set and show probabilities for each class.
i = np.random.randint(0,len(X_test))
digit = X_test[i].reshape(28,28)

plt.figure();
plt.subplot(1,2,1);
plt.title('Example of digit: {}'.format(y_test[i]));
plt.imshow(digit,cmap='gray'); plt.axis('off');
probs = model.predict_proba(digit.reshape(1,1,28,28),batch_size=1)
plt.subplot(1,2,2);
plt.title('Probabilities for each digit class');
plt.bar(np.arange(10),probs.reshape(10),align='center'); plt.xticks(np.arange(10),np.arange(10).astype(str));

predictions = model.predict_classes(X_test, batch_size=32, verbose=1)

inds = np.arange(len(predictions))
wrong_results = inds[y_test!=predictions]

#choose a random wrong result from the test set
i = np.random.randint(0,len(wrong_results))
i = wrong_results[i]
digit = X_test[i].reshape(28,28)

plt.figure();

plt.subplot(1,2,1);
plt.title('Digit {}'.format(y_test[i]));
plt.imshow(digit,cmap='gray'); plt.axis('off');
probs = model.predict_proba(digit.reshape(1,1,28,28),batch_size=1)
plt.subplot(1,2,2);
plt.title('Digit classification probability');
plt.bar(np.arange(10),probs.reshape(10),align='center'); plt.xticks(np.arange(10),np.arange(10).astype(str));

prediction_probs = model.predict_proba(X_test, batch_size=32, verbose=1)

wrong_probs = np.array([prediction_probs[ind][digit] for ind,digit in zip(wrong_results,predictions[wrong_results])])
all_probs =  np.array([prediction_probs[ind][digit] for ind,digit in zip(np.arange(len(predictions)),predictions)])
#plot as histogram
plt.hist(wrong_probs,alpha=0.5,normed=True,label='wrongly-labeled');
plt.hist(all_probs,alpha=0.5,normed=True,label='all labels');
plt.legend();
plt.title('Comparison between wrong and correctly classified labels');
plt.xlabel('highest probability');

for i,arr in enumerate(model.layers[0].get_weights()[0]):
    plt.subplot(6,6,i+1)
    plt.imshow(arr[0,:,:],cmap='gray',interpolation='none'); plt.axis('off');

#Create new sequential model, same as before but just keep the convolutional layer. 
model_new = Sequential()

model_new.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model_new.add(Activation('relu'))

#set weights for new model from weights trained on MNIST.
for i in range(2):
    model_new.layers[i].set_weights(model.layers[i].get_weights())

#pick a random digit and "predict" on this digit (output will be first layer of CNN)
i = np.random.randint(0,len(X_test))
digit = X_test[i].reshape(1,1,28,28)
pred = model_new.predict(digit)

#For all the filters, plot the output of the input
plt.figure(figsize=(18,18))
filts = pred[0]
for i,fil in enumerate(filts):
    plt.subplot(6,6,i+1)
    plt.imshow(fil.reshape(26,26),cmap='gray'); plt.axis('off');



