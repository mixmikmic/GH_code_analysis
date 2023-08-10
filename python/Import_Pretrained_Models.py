from keras.applications import InceptionV3, ResNet50
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import Model

orig = InceptionV3(weights='imagenet', )

# orig.summary()

model = InceptionV3(input_shape=(480, 480, 3), weights='imagenet', include_top=False)

# model.summary()

last_layer_output = model.layers[-1].output
x = GlobalAveragePooling2D()(last_layer_output)
x = Dense(128, activation='softmax', name='predictions')(x)

new_model = Model(inputs=model.input, outputs=x)

# new_model.summary()

for l in new_model.layers[-20:]:
    if l.name.startswith('conv'):
        print l.name
        l.trainable = False

# new_model.summary()



