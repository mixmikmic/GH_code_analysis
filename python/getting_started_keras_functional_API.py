from keras.layers import Input,Dense
from keras.models import Model

# For a single-input model with 2 classes (binary classification):
# Generate dummy data
import numpy as np
np.random.seed(10)
data_train = np.random.random((1000, 100))
labels_train = np.random.randint(2, size=(1000, 1))
data_test = np.random.random((200, 100))
labels_test = np.random.randint(2, size=(200, 1))

# This returns a tensor
inputs = Input(shape=(100,))
x = Dense(32, activation='relu')(inputs)
predictions = Dense(1, activation='sigmoid')(x)

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions) 

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model, iterating on the data in batches of 32 samples
model.fit(data_train, labels_train, epochs=100, batch_size=32)

# Evaluating the performance of the trained model on test data
score = model.evaluate(data_test,labels_test)
print ('\nAccuracy: ',score[0]*100,'%')
print ('Loss: ',score[1])

# predicting the output of a single test data
pred = model.predict(data_test[6:7]) # the output is the probablity of predicting 1 since its a binary classification
if pred>0.5:
    pred_value = 1
else:
    pred_value = 0
true = labels_test[6]
print ('predicted_probability: ',pred)
print ('predicted label: ',pred_value)
print ('true label: ',true)

# saving the trained model
model.save('my_model_functional_api.hdf5')



