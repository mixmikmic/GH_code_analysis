from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.applications as apps

model = apps.densenet.DenseNet201(include_top=True, weights='imagenet', classes=1000)

model.summary()

image = load_img('husky.jpg', target_size=(224, 224))
image = img_to_array(image)

plt.imshow(np.uint8(image))

# Preprocessing the image
image_processed = apps.densenet.preprocess_input(image)
image_processed = image_processed.reshape((1, 224, 224, 3))

# Get the predictions
from time import time
start_time = time()
predictions = model.predict([image_processed])
end_time = time()
print("feed forward time = ", (end_time-start_time), "s")
print("Predictions shape:", predictions.shape)


# Keras also provides a handy decoding utility
labels = apps.densenet.decode_predictions(predictions)[0]

for _, label, prob in labels:
    print(label, " (prob:", prob*100, "%)")

