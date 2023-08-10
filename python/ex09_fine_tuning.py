from keras.applications import VGG16
conv_base=VGG16(weights="imagenet",
               include_top=False,
               input_shape=(150,150,3))

from keras import layers
from keras import models

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()

print("Before freezing, trainable params num: "+str(len(model.trainable_weights)))
conv_base.trainable=False
print("After freezing, trainable params num: "+str(len(model.trainable_weights)))

from keras.preprocessing.image import ImageDataGenerator
import os
train_gen=ImageDataGenerator(rescale=1/255.,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode="nearest")
test_gen=ImageDataGenerator(rescale=1/255.)

def getGen(gen,dir_):
    return gen.flow_from_directory(dir_,
                                  target_size=(150,150),
                                  batch_size=20,
                                  class_mode="binary")
train_dir=os.path.join("small_train","train")
val_dir=os.path.join("small_train","val")
train_gener=getGen(train_gen,train_dir)
val_gener=getGen(test_gen,val_dir)

from keras import optimizers
model.compile(
            #optimizer="rmsprop",
             optimizer=optimizers.RMSprop(lr=2e-5),
             loss="binary_crossentropy",
             metrics=["acc"])
history=model.fit_generator(train_gener,
                            steps_per_epoch=100,
                            epochs=20,
                            validation_data=val_gener,
                            validation_steps=50)

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=="block5_conv1":
        set_trainable=True
    layer.trainable=set_trainable
    

history=model.fit_generator(train_gener,
                           steps_per_epoch=100,
                           epochs=10,
                           validation_data=val_gener,
                           validation_steps=50)

test_dir=os.path.join("small_train","test")
test_gener=getGen(test_gen,test_dir)
loss,acc=model.evaluate_generator(test_gener)

print("Test loss: "+str(loss)+". Test acc: "+str(acc))



