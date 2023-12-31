import keras
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

# inception v3 model
model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
graph = tf.get_default_graph()

# Prediction function definition
def predict(image_file):
    img = image.load_img(image_file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    global graph
    with graph.as_default():
        preds = model.predict(x)

    top3 = decode_predictions(preds,top=3)[0]

    predictions = [{'label': label, 'description': description, 'probability': probability * 100.0}
                    for label,description, probability in top3]
    return predictions


image_file ="./data/Beach.jpg"
prediction = predict(image_file)

for i in range(0,len(prediction)):
    print((i+1))
    print('Description:',prediction[i]['description'],'\nProablility:',prediction[i]['probability'],'\n')

