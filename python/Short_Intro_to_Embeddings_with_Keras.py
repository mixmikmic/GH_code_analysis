import numpy as np

embedding_size = 4
vocab_size = 10

embedding = np.arange(embedding_size * vocab_size, dtype='float')
embedding = embedding.reshape(vocab_size, embedding_size)
print(embedding)

i = 3
onehot = np.zeros(10)
onehot[i] = 1.
onehot

embedding_vector = np.dot(onehot, embedding)
print(embedding_vector)

print(embedding[i])

from tensorflow.contrib import keras
from keras.layers import Embedding

embedding_layer = Embedding(
    output_dim=embedding_size, input_dim=vocab_size,
    input_length=1, name='my_embedding')

from keras.layers import Input
from keras.models import Model

x = Input(shape=[1], name='input')
embedding = embedding_layer(x)
model = Model(input=x, output=embedding)
model.output_shape

model.get_weights()

model.predict([[0],
               [3]])

from keras.layers import Flatten

x = Input(shape=[1], name='input')

# Add a flatten layer to remove useless "sequence" dimension
y = Flatten()(embedding_layer(x))

model2 = Model(input=x, output=y)
model2.output_shape

model2.predict([[0],
                [3]])



