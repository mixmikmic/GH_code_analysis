import pandas as pd

lm='/home/ubuntu/list_landmarks-2.txt'
text=pd.read_csv(lm,delim_whitespace=True,skiprows=0,header=1)


text.head()

text.columns

text.describe()

# In clothes type, "1" represents upper-body clothes, 
# Landmarks : ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"];
upper=text[text.clothes_type ==1]


upper.describe()

#In landmark visibility state, "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off;

upper.describe()

# testing - invisible collar ? 
t1=upper[upper.landmark_visibility_1==1]
t1.head()

# testing - occluded left sleeve? 
t2=upper[upper.landmark_visibility_3==1]
t2.head()

from scipy import ndimage
from matplotlib import pyplot as plt


file1=ndimage.imread('/home/ubuntu/img/img_00000001.jpg')
plt.imshow(file1)
plt.show()

# "2" represents truncated/cut-off; going to leave those out.. 
upper2=upper[(upper.landmark_visibility_1!=2)&(upper.landmark_visibility_2!=2)&(upper.landmark_visibility_3!=2)&
            (upper.landmark_visibility_4!=2) & (upper.landmark_visibility_5!=2) & (upper.landmark_visibility_6!=2)]

upper2.describe()

from scipy import ndimage

file1='/home/ubuntu/list_eval_partition-3.txt'

split=pd.read_csv(file1,delim_whitespace=True,skiprows=[0])

split.head()
len(split)
#len(upper2)

len(upper2)

splits=upper2.merge(split,how='left')

splits.head(1)

len(splits)

split.head()

# create new column for just image name

def image(x):
    x=x.split('/')[1:]
    return '/'.join(x)

splits['image']=splits.image_name.apply(image)

def filename(df):
    x=df.evaluation_status+'/'+df.image
    return x

splits['file']=splits.apply(filename,axis=1)
splits.head()



test=splits[100:200]

test.head()

# testing small sample :O
import os
from shutil import move
from shutil import copyfile


for i in test.index:
    my_file='/home/ubuntu/'+test.image_name[i]
    # move file to upper
    #new_file='/home/ubuntu/upper/'+test.evaluation_status[i]+'/'+test.image[i]
    new_path='/home/ubuntu/upper/'+test.evaluation_status[i]
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    #try:
    #print new_path
    print my_file
    move(my_file,new_path)
    #! mv my_file new_file
    #except:
        #print new_path,my_file, new_file
        
    
    

for i in splits.index:
    my_file='/home/ubuntu/'+splits.image_name[i]
    # move file to upper
    #new_file='/home/ubuntu/upper/'+test.evaluation_status[i]+'/'+test.image[i]
    new_path='/home/ubuntu/upper/'+splits.evaluation_status[i]
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    #try:
    #print new_path
    #print my_file
    try:
        move(my_file,new_path)
    except:
        print my_file
    #! mv my_file new_file
    #except:
        #print new_path,my_file, new_file

import os
from shutil import copyfile

rejects=[]

for i in splits.index:
    my_file='/home/ubuntu/'+splits.image_name[i]
    new_file='/home/ubuntu/upper/'+splits.evaluation_status[i]+'/'+splits.image[i]
    new_path='/home/ubuntu/upper/'+splits.evaluation_status[i]
    #if not os.path.exists(new_file):
        #os.makedirs(new_file)
    try:
        move(my_file,new_path)
    except:
        rejects.append(my_file)
    

# preprocesing on targets:

# a) change the  0's to 1's (ease for working with the loss function)
# b) normalize landmarks for loss ? 

upper2.head()

upper2.describe()

subs=upper2[['landmark_visibility_1','landmark_visibility_2','landmark_visibility_3','landmark_visibility_4',
            'landmark_visibility_5','landmark_visibility_6']]
subs.describe()

upper2.columns



# Preprocess : 

subs=upper2[['landmark_visibility_1','landmark_visibility_2','landmark_visibility_3','landmark_visibility_4',
            'landmark_visibility_5','landmark_visibility_6']]

subs=subs.replace({1:0,0:1})

subs.describe()

# update each.. 
upper2.landmark_visibility_6=subs.landmark_visibility_6
upper2.describe()

## testing 

upper2[(upper2.landmark_visibility_1==0) & (upper2.landmark_visibility_2==0)&(upper2.landmark_visibility_4==0) &       (upper2.landmark_visibility_3==0)&(upper2.landmark_visibility_5==0) & (upper2.landmark_visibility_6==0)]

from scipy import ndimage
from matplotlib import pyplot as plt


file1=ndimage.imread('/home/ubuntu/img/img_00123059.jpg')
plt.imshow(file1)
plt.show()

upper2.head()

upper2.to_csv('upper_files.csv')

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    return model



loc = VGG_16()
loc.add(Dense(8,activation='linear',name='location'))

vis1=VGG_16()
vis1.add(Dense(2,activation='softmax',name='visibility1'))


from keras import metrics


sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=[metrics.top_k_categorical_accuracy])

from keras.preprocessing import image

train_gen= keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())

train_generator = train_gen.flow_from_directory(
        '/home/ubuntu/mount_point/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_datagen = keras.preprocessing.image.ImageDataGenerator()
validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/mount_point/test',
        target_size=(224,224),
        batch_size=32)

# Refine for our application
model.layers.pop()
model.add(Dropout(0.5))
model.add(Dense(5596))
model.add(Activation('softmax'))

sgd=SGD(lr=.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=[metrics.top_k_categorical_accuracy])

# GPU is about 100 x faster than non GPU, I guess all of this effort was worth it~ (started 3:30?)
# 10937 gpu usage
model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.nb_sample,
        nb_epoch=10,
        nb_val_samples=validation_generator.nb_sample,
        validation_data=validation_generator)
model.save('my_weights.h5')

model.save('my_weights.h5')





