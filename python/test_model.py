get_ipython().magic('matplotlib inline')
import numpy as np

from keras.models import model_from_json
from skimage.io import imread, imshow
from skimage.transform import resize

json_file = open('model.json','r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("model.h5")
print("Loaded Model from disk")

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adadelta',
             metrics=['accuracy'])

x = imread('test.png')
x = np.invert(x)
x = resize(x,(28,28))
imshow(x[:,:,1])

x = x[:,:,1].reshape(1,28,28,1)

out = model.predict(x)
print(out)

print(np.argmax(out, axis=1))



