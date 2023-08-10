from IPython.display import Image
Image("img/1.png")

Image("img/2.png")

import os
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


from keras.preprocessing.image import  img_to_array
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')




import numpy as np


from PIL import Image


from sklearn.cross_validation import train_test_split
ROOT_PATH = ""
train_data_dir = os.path.join(ROOT_PATH, "datasets/Training/")
test_data_dir = os.path.join(ROOT_PATH, "datasets/Testing/")
print(train_data_dir)

m,n = 50,50


classes=os.listdir(train_data_dir)
x=[]
y=[]
for fol in classes:
#     print (fol)
    imgfiles=os.listdir(train_data_dir+fol);
    for img in imgfiles :
        
        if img.endswith(".ppm"):
            im=Image.open(train_data_dir+fol+'/'+img);
            im=im.convert(mode='RGB')
            imrs=im.resize((m,n))
            imrs=img_to_array(imrs)/255;
            imrs=imrs.transpose(2,0,1);
            imrs=imrs.reshape(3,m,n);
            x.append(imrs)
            y.append(fol)

print(x[0])
x=np.array(x)
y=np.array(y)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
# print(x_train[0])





batch_size=32
nb_classes=len(classes)
nb_epoch=5
nb_filters=32
nb_pool=2
nb_conv=3


uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)



model= Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])  

model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
model.save('traffic.h5')
model.save_weights('my_model_weights.h5')
json_string = model.to_json()
with open('traffic.json','w') as f:
    f.write(json_string)
    

path1='datasets/Try/'
files=os.listdir(path1)
img=files[0]
im = Image.open(path1 +img)
imrs = im.resize((m,n))
imrs=img_to_array(imrs)/255
imrs=imrs.transpose(2,0,1)
imrs=imrs.reshape(3,m,n)

x=[]
x.append(imrs)
x=np.array(x)

preds= model.predict(x)[0]
print (preds)
print (preds[0])
print(np.argmax(preds))

def load_data(data_dir):
  
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
       
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels
images, labels = load_data(train_data_dir)
def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

display_images_and_labels(images, labels)
#for simple reference go to : https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6



