from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

# Use whole vgg16 model
# Input image format: (224 X 224 X 3)
vgg16_with_top = VGG16(include_top=True, weights='imagenet',
                                input_tensor=None, input_shape=None,
                                pooling=None,
                                classes=1000)
vgg16_with_top.summary()

# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

# Stop to train weights of convolutional layers, if you want to fit them, markdown the following two lines
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Create your own input format (here 224 X 224 X 3)
inputs = Input(shape=(224,224,3),name = 'image_input')

# Use the generated model 
output_vgg16_conv = model_vgg16_conv(inputs)

# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
#x = Dense(128, activation='relu', name='fc1')(x)
#x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(1000, activation='softmax', name='predictions')(x)

# Create your own model 
my_model = Model(inputs=inputs, output=x)

# In the summary, weights and layers from VGG part will be hidden, and they will not be fit during traning
my_model.summary()

# Generate a model with all layers (with top)
my_vgg16_model = VGG16(weights='imagenet', include_top=True)

# Stop to train weights of VGG16 layers
for layer in my_vgg16_model.layers:
    layer.trainable = False
    
# Add a layer where input is the output of the second last layer
#x = Flatten(name='flatten')(my_vgg16_model.layers[-2].output)

# Add a layer where input is the output of the fourth last layer
x = Dropout(0.9, noise_shape=None, seed=None)(my_vgg16_model.layers[-4].output) 
#x = Dense(128, activation='relu', name='fc1')(x)
#x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(1000, activation='softmax', name='predictions')(x)

# Then create the corresponding model 
my_model = Model(inputs=my_vgg16_model.input, output=x)
my_model.summary()

