import os    
#os.environ['THEANO_FLAGS'] = "device=gpu1"  
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
import theano
import numpy as np
from keras.layers import Input, Dense, convolutional,Reshape, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.optimizers import *

get_ipython().magic('matplotlib inline')





# this is our input placeholder
input_img = Input(shape=(3,32,32))

x = convolutional.Convolution2D(16, 5, 5, activation='relu', border_mode='same')(input_img)##64, 28, 28
x = convolutional.MaxPooling2D((2, 2), border_mode='same')(x)##64, 14, 14
x = convolutional.Convolution2D(20, 5, 5, activation='relu', border_mode='same')(x) ##32, 12, 12
x = convolutional.MaxPooling2D((2, 2), border_mode='same')(x) ##32, 6, 6
x = convolutional.Convolution2D(20, 5, 5, activation='relu', border_mode='same')(x) ##16, 4, 4
#x = convolutional.MaxPooling2D((2, 2), border_mode='same')(x) ##32, 6, 6
#x = convolutional.Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x) ##16, 4, 4
encoded = convolutional.MaxPooling2D((2, 2), border_mode='same')(x) ##16, 2, 2
#x = Flatten()(encoded) ##64
#encoded = Dense(40, activation='sigmoid')(x)
#x= Dense (64, activation = 'linear')(x)
#x = Reshape((16,2,2))(x) ##128, 4, 4
x = convolutional.UpSampling2D((2, 2))(encoded) 
#x = convolutional.Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x) ##16, 4, 4
#x = convolutional.UpSampling2D((2, 2))(x) 
x = convolutional.Convolution2D(20, 5, 5, activation='relu', border_mode='same')(x) 
x = convolutional.UpSampling2D((2, 2))(x) 
x = convolutional.Convolution2D(20, 5, 5, activation='relu', border_mode='same')(x) 
x = convolutional.UpSampling2D((2, 2))(x)
x = convolutional.Convolution2D(16, 5, 5, activation='relu',border_mode='same')(x) ##32, 16, 16
decoded = convolutional.Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x) ##3, 32, 32
#x = Flatten()(x) ##3072
#x = Dense(3072, activation='linear')(x)
#decoded = Reshape((3,32,32))(x) ##3, 32, 32
#decoded = convolutional.Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)
autoencoder = Model(input_img, decoded)
#sgd=SGD(lr=0.002, momentum=0.1, decay=0.0, nesterov=False)
autoencoder.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

autoencoder.summary()

X_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_train.npy')
X_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_test.npy')
y_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_train.npy')
y_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_test.npy')

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)



x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

x_train = x_train.reshape((len(x_train),3,32,32))
x_test = x_test.reshape((len(x_test),3,32,32))
print (x_train.shape)
print (x_test.shape)

# this model maps an input to its encoded representation
#encoding_dim=40
#encoded = convolutional.Convolution2D(3, 5, 5, activation='relu', border_mode='same')(encoded)
encoded = Flatten()(encoded)
#encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
#encoded = convolutional.Convolution2D(3, 5, 5, activation='relu', border_mode='same')(encoded)
#encoded = Reshape((3,32,32))(encoded) ##3, 32, 32

encoder = Model(input_img, encoded)

print (encoder.summary())

nb_epoch=10
batch_size=64
random_state =5578

kfold_weights_path = os.path.join('weights_kfold_' +  'CIFAR10-Conv-AutoEncoder-Ver6' +
                                  '_epoch_'+str(nb_epoch)+
                                  '_batch_'+str(batch_size)
                                  +'.h5')
print(kfold_weights_path)

os.path.isfile(kfold_weights_path)

# Some transfer learning
if os.path.isfile(kfold_weights_path):
    print ('Loading already stored weights...')
    autoencoder.load_weights(kfold_weights_path)
else:
    print ('Training for the first time...')
    

callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=1),
            ]
autoencoder.fit(x_train, x_train,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=callbacks
            )

autoencoder.summary()

reconstructed_test_imgs = autoencoder.predict(x_test)

reconstructed_test_imgs.shape

import matplotlib.pyplot as plt

n = 15  # how many digits we will display
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded
    #ax = plt.subplot(2, n, i + 1 + n)
    #plt.imshow(encoded_imgs[i].reshape(6, 6))
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
   
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_test_imgs[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print (encoder.summary())

encoded_train_imgs = encoder.predict(x_train)
print (encoded_train_imgs.size)
print (encoded_train_imgs.shape)
print (encoded_train_imgs.nbytes)

encoded_test_imgs = encoder.predict(x_test)
print (encoded_test_imgs.size)
print (encoded_test_imgs.shape)
print (encoded_test_imgs.nbytes)

encoded_train_imgs

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)
clf = clf.fit(encoded_train_imgs, y_train)

y_pred = clf.predict(encoded_test_imgs)

y_pred

num=len(encoded_test_imgs)
r=0
w=0
for i in range(num):
        #print ('y_pred ',y_pred[i])
        #print ('labels ',labels[i])
        #without the use of all() returns error truth value of an array with more than one element is ambiguous
        #if y_pred[i].all() == labels[i].all():
        if np.array_equal(y_pred[i],y_test[i]):
            r+=1
        else:
            w+=1
print ("tested ",  num, "digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")

input_dim = Input(shape=(320,))
x = Dense(100, activation='relu')(input_dim)
classifier = Dense(10, activation='softmax')(x)
nn = Model(input=input_dim, output=classifier)

nn.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

from keras.utils.np_utils import *

label_train=to_categorical(y_train)
label_test= to_categorical(y_test)
print (label_train.shape)
print (label_test.shape)

nn.fit(encoded_train_imgs, label_train,
            nb_epoch=100,
            batch_size=batch_size,
            shuffle=True)

label_y_pred = nn.predict(encoded_test_imgs)

new_y_pred=[]
for i in range (len(label_y_pred)):
    new_y_pred.append([np.argmax(label_y_pred[i])])

new_y_pred=np.asarray(new_y_pred)

print (new_y_pred.shape)
new_y_pred=to_categorical(new_y_pred)
print (new_y_pred.shape)

num=len(label_y_pred)
r=0
w=0
for i in range(num):
        #print ('y_pred ',y_pred[i])
        #print ('labels ',labels[i])
        #without the use of all() returns error truth value of an array with more than one element is ambiguous
        #if y_pred[i].all() == labels[i].all():
        if np.array_equal(new_y_pred[i],label_test[i]):
            r+=1
        else:
            w+=1
print ("tested ",  num, "digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")

from sklearn import linear_model
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().magic('matplotlib inline')
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA

print(y_train.shape)
y_train_pd=y_train.reshape(50000,1)
print(y_train_pd.shape)
print (y_test.shape)
y_test_pd=y_test.reshape(10000,1)

print (y_test_pd.shape)

from sklearn.manifold import TSNE
X = encoded_test_imgs
print (X.shape)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
t_sne_X =model.fit_transform(X) 
print (t_sne_X.shape)

t_sne_test = pd.DataFrame(t_sne_X,columns=list('AB'))
t_sne_test['Label'] = pd.DataFrame(y_test_pd)
t_sne_test.head()

test_df_sample=t_sne_test[0:30]
test_df_sample=test_df_sample[test_df_sample.Label < 10]

sb.set_context("notebook", font_scale=1.1)
sb.set_style("ticks")


sb.lmplot('A','B',
           data=test_df_sample, 
           fit_reg=False, 
           hue="Label",  
           scatter_kws={"marker": "Label", 
                        "s": 100})
plt.title('Plot T-Sne')
plt.xlabel('T-SNE#1')
plt.ylabel('T-SNE#2')

reduced_train_data = PCA(n_components=2).fit_transform(encoded_train_imgs)
reduced_test_data = PCA(n_components=2).fit_transform(encoded_test_imgs)

print (reduced_train_data.shape)
print (reduced_test_data.shape)

train_df = pd.DataFrame(reduced_train_data)

train_df['Label'] = pd.DataFrame(y_train_pd)

train_df.head()

test_df = pd.DataFrame(reduced_test_data,columns=list('AB'))
test_df['Label'] = pd.DataFrame(y_test_pd)
test_df.head()

# Create 
kmeans = KMeans(n_clusters=10)
kmeans.fit(reduced_train_data) 

kmeans.cluster_centers_.shape

kmeans.labels_.shape

km_predicted = kmeans.predict(reduced_test_data)
cluster= km_predicted
print (cluster.shape)
cluster=cluster.reshape(10000,1)
print (cluster.shape)

np.unique(cluster)

np.unique(test_df.Label)

test_df['Cluster'] = pd.DataFrame(cluster)
test_df.head(10)



test_df_sample=test_df[0:10000]
test_df_sample=test_df_sample[test_df_sample.Label < 3]

sb.set_context("notebook", font_scale=1.1)
sb.set_style("ticks")


sb.lmplot('A','B',
           data=test_df_sample, 
           fit_reg=False, 
           hue="Label",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Histogram of IQ')
plt.xlabel('PC#1')
plt.ylabel('PC#2')

test_df_sample=test_df[0:10000]
test_df_sample=test_df_sample[test_df_sample.Cluster < 3]

sb.set_context("notebook", font_scale=1.1)
sb.set_style("ticks")


sb.lmplot('A','B',
           data=test_df_sample, 
           fit_reg=False, 
           hue="Cluster",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Histogram of IQ')
plt.xlabel('PC#1')
plt.ylabel('PC#2')

cmap = sb.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
g = sb.clustermap(test_df[0:2], cmap=cmap, linewidths=.5)

reduced_train_data = reduced_train_data[0:501,]

reduced_train_data.shape

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .05    # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_train_data[:, 0].min() - 1, reduced_train_data[:, 0].max() + 1
y_min, y_max = reduced_train_data[:, 1].min() - 1, reduced_train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_train_data[:, 0], reduced_train_data[:, 1], 'k.', markersize=7)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

raw_data['Cluster_Label'] = kmeans.labels_

