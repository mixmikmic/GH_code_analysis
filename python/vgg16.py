import PIL
import requests
import numpy as np
from io import BytesIO
from IPython.display import Image

import tensorflow as tf
from tensorflow.contrib.keras.python.keras import applications
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.applications import imagenet_utils

vgg16 = applications.VGG16(weights="imagenet")

vgg16.summary()

url = "https://media4.s-nbcnews.com/j/newscms/2016_36/1685951/ss-160826-twip-05_8cf6d4cb83758449fd400c7c3d71aa1f.nbcnews-ux-2880-1000.jpg"

Image(url, width=224, height=224)

response = requests.get(url)

img = image.load_img(BytesIO(response.content), target_size=(224, 224))

img

arr = image.img_to_array(img)

arr.shape

queries = np.expand_dims(arr, axis=0)

queries = imagenet_utils.preprocess_input(queries)

predictions = vgg16.predict(queries)

predictions[0, :5]

imagenet_utils.decode_predictions(predictions)



