import PIL
import requests
import numpy as np
from io import BytesIO
from IPython.display import Image

import tensorflow as tf

## Use base keras instead of tf.keras because of: https://github.com/tensorflow/tensorflow/issues/11868
# from tensorflow.contrib.keras.python.keras import applications
# from tensorflow.contrib.keras.python.keras.preprocessing import image
# from tensorflow.contrib.keras.python.keras.applications import imagenet_utils
##
from keras import applications
from keras.preprocessing import image
from keras.applications import imagenet_utils
##

resnet50 = applications.ResNet50(weights="imagenet")

resnet50.summary()

url = "https://www.euroresidentes.com/suenos/img_suenos/caballo-suenos-euroresidentes.jpg"

Image(url, width=224, height=224)

response = requests.get(url)

img = image.load_img(BytesIO(response.content), target_size=(224, 224))

img

arr = image.img_to_array(img)

arr.shape

queries = np.expand_dims(arr, axis=0)

queries = imagenet_utils.preprocess_input(queries)

predictions = resnet50.predict(queries)

predictions[0, :5]

imagenet_utils.decode_predictions(predictions)



