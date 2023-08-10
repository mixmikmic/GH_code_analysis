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


from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense

from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.regularizers import *
from keras.layers.normalization import *

x_train = np.genfromtxt('x_train.out')
y_train = np.genfromtxt('y_train.out')
vx_train = np.genfromtxt('vx_train.out')
vy_train = np.genfromtxt('vy_train.out')
x_test = np.genfromtxt('x_test.out')
y_test = np.genfromtxt('y_test.out')

print (x_train.shape)
print (y_train.shape)
print (vx_train.shape)
print (vy_train.shape)
print (x_test.shape)
print (y_test.shape)

train_x_temp = x_train.reshape(-1,1, 28, 28)
val_x_temp = vx_train.reshape(-1,1, 28, 28)
test_x_temp=x_test.reshape(-1,1, 28, 28)
print(train_x_temp.shape)
print((train_x_temp.dtype))

print(val_x_temp.shape)
print(val_x_temp.dtype)

print(test_x_temp.shape)
print(test_x_temp.dtype)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

datagen

# fit parameters from data
# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_x_temp)
#datagen_fit_v=datagen.fit(val_x_temp)

import random

ntrain = 50000
print (train_x_temp.shape[0])
trainidx = random.sample(range(0,train_x_temp.shape[0]), ntrain)
print (len(trainidx))
XT = train_x_temp[trainidx,:,:,:]
print (XT.shape)

ntrain_v = 10000
print (val_x_temp.shape[0])
validx = random.sample(range(0,val_x_temp.shape[0]), ntrain_v)
print (len(validx))
XV = val_x_temp[validx,:,:,:]
print (XV.shape)

i=0
generated_images = np.empty(shape=[0,1,28,28])
print (generated_images.shape)
for X_batch in datagen.flow(XT,batch_size=500):
    generated_images = np.concatenate((generated_images, X_batch))
    i += 1
    if i > 99:
        break  # otherwise the generator would loop indefinitely

i=0
generated_images_v = np.empty(shape=[0,1,28,28])
print (generated_images_v.shape)
for V_batch in datagen.flow(XV,batch_size=100):
    generated_images_v = np.concatenate((generated_images_v, V_batch))
    i += 1
    if i > 99:
        break  # otherwise the generator would loop indefinitely

print (generated_images.shape)
print (generated_images_v.shape)

import matplotlib.pyplot as pyplot

for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(XT[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
	# show the plot
pyplot.show()

for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(generated_images[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()

for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(generated_images_v[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()

X = np.concatenate((XT, generated_images))
print (X.shape)
n = XT.shape[0]
y = np.zeros([2*n,2])
print (y.shape)
y[:n,1] = 1
y[n:,0] = 1

y

val_X=np.concatenate((XV, generated_images_v))
print (val_X.shape)
n = XV.shape[0]
val_y = np.zeros([2*n,2])
print (val_y.shape)
val_y[:n,1] = 1
val_y[n:,0] = 1
print (val_y)

shp = XT.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)
print (shp)

# Build Discriminative model ...
d_input = Input(shape=shp)
H = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(128, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(128)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
#encoder = Dense(40,activation='sigmoid')(H)--not gving accuracy?what is the default?
encoder = Dense(40)(H)
d_V = Dense(2,activation='softmax')(encoder)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

encoder = Model(d_input,encoder)
encoder.compile(loss='categorical_crossentropy', optimizer=dopt)
encoder.summary()

nb_epoch=50
batch_size=128
random_state =55789

kfold_weights_path = os.path.join('weights_kfold_' +  'MNIST-Aug_Discmtv_Conn_Encoder-Ver1.3' +
                                  '_epoch_'+str(nb_epoch)+
                                  '_batch_'+str(batch_size)
                                  +'.h5')
print(kfold_weights_path)

os.path.isfile(kfold_weights_path)

# Some transfer learning
if os.path.isfile(kfold_weights_path):
    print ('Loading already stored weights...')
    discriminator.load_weights(kfold_weights_path)
else:
    print ('Training for the first time...')
    

callbacks = [
                EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=1),
            ]
history=discriminator.fit(X, y,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(val_X, val_y),
            callbacks=callbacks
            )

print(history.history.keys())
# summarize history for accuracy
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()

# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()



encoded_train_imgs = encoder.predict(train_x_temp)
print (encoded_train_imgs.size)
print (encoded_train_imgs.shape)
print (encoded_train_imgs.nbytes)

encoded_test_imgs = encoder.predict(test_x_temp)
print (encoded_test_imgs.size)
print (encoded_test_imgs.shape)
print (encoded_test_imgs.nbytes)

encoded_test_imgs

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(encoded_train_imgs, y_train)

y_pred = clf.predict(encoded_test_imgs)

y_pred

y_test

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
# With Val only having Real images 5 epochs - 48.17%
# With Val also having generated images +5 epochs - 28.89%
# With Model complexity reduced and 5 epochs - 82.4%
# With Model complexity reduced and 5 epochs - 81.4%
# With1.1 ZCA added and 50 epochs - 64.61%
# With1.1 ZCA removed and 50 epochs - 82.86%

from sklearn import linear_model
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().magic('matplotlib inline')
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA

# apply pca to reduce to 2 dimensions
reduced_train_data = PCA(n_components=2).fit_transform(encoded_train_imgs)
reduced_test_data = PCA(n_components=2).fit_transform(encoded_test_imgs)

# dont appy and further DR
reduced_train_data = encoded_train_imgs
reduced_test_data = encoded_test_imgs

print (reduced_train_data.shape)
print (reduced_test_data.shape)

print(y_train.shape)
y_train_pd=y_train.reshape(50000,1)
print(y_train_pd.shape)
print (y_test.shape)
y_test_pd=y_test.reshape(10000,1)

print (y_test_pd.shape)

train_df = pd.DataFrame(reduced_train_data)

train_df['Label'] = pd.DataFrame(y_train_pd)

train_df.head()

#test_df = pd.DataFrame(reduced_test_data,columns=list('AB'))
test_df = pd.DataFrame(reduced_test_data)
test_df['Label'] = pd.DataFrame(y_test_pd)
test_df.head()

# Create kmeans
clstr = KMeans(n_clusters=10)
clstr.fit(reduced_train_data) 

clstr.cluster_centers_.shape

clstr.labels_.shape

clstr_predicted = clstr.predict(reduced_test_data)
#clstr_predicted = clstr.fit_transform(reduced_test_data) 

cluster= clstr_predicted
print (cluster.shape)
cluster=cluster.reshape(10000,1)
print (cluster.shape)

np.unique(cluster)

np.unique(test_df.Label)

test_df['Cluster'] = pd.DataFrame(cluster)
test_df.head(10)

test_df['Cluster'].hist(by=test_df['Label'])

test_df.groupby(['Label','Cluster'])['Cluster'].count()

test_df.groupby(['Cluster','Label'])['Label'].count()

test_df.groupby(['Cluster'])['Label'].count()

#Determine Cluster accuracy
from scipy.stats import mode 

f = lambda x: mode(x, axis=None)[0]

test_df.groupby(['Cluster'])['Label'].apply(f)



#Actual_Label = test_df['Label'].tolist()
#Cluster_Label = test_df['Cluster'].tolist()
Actual_Label = np.array(test_df['Label'])
Cluster_Label = np.array(test_df['Cluster'])

Actual_Label

Cluster_Label

Assigned_labels=np.zeros_like(Cluster_Label)
#Assigned_labels = Assigned_labels.tolist()
#Assigned_labels

Cluster_Label==1

for i in range(10):
    print(i)
    mask=(Cluster_Label==i)
    #print(mask)
    Assigned_labels[mask]= mode(Actual_Label[mask])[0]
    print(len(Actual_Label[mask]))
    print(mode(Actual_Label[mask])[0])
    print(len(Assigned_labels[mask]))
    #print(labels[i])
print(np.array(Assigned_labels))

Assigned_labels=np.array(Assigned_labels)

from sklearn.metrics import accuracy_score
accuracy_score(Assigned_labels,Actual_Label)

Assigned_labels.shape

Actual_Label.shape

test_df.head()

test_df.describe()

test_df.groupby('Cluster').count()

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
test_df_sample=test_df_sample[test_df_sample.Cluster < 10]

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

