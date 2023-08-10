import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet', include_top=True)

layers = dict([(layer.name, layer.output) for layer in model.layers])
layers

model.count_params() # A lot!

image_path = 'images/elephant.jpg'
image = Image.open(image_path)
image = image.resize((224, 224))
image

# Convert it into an array
x = np.asarray(image, dtype='float32')
# Convert it into a list of arrays
x = np.expand_dims(x, axis=0)
# Pre-process the input to match the training data
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

from keras.models import Model
base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

block4_pool_features = model.predict(x)
print(block4_pool_features)
print(block4_pool_features.shape)

