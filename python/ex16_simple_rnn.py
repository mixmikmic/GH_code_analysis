from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=10000
maxlen=20
batch_size=32

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

train_data=sequence.pad_sequences(x_train,maxlen=maxlen)
test_data=sequence.pad_sequences(x_test,maxlen=maxlen)

train_data.shape

from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN,Dense

model=Sequential()
model.add(Embedding(max_features,100,input_length=maxlen))
model.add(SimpleRNN(32))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="rmsprop",
             loss="binary_crossentropy",
             metrics=["acc"])

model.summary()

history=model.fit(train_data,y_train,
                 epochs=10,
                 batch_size=batch_size,
                 validation_split=0.2)

model.evaluate(test_data,y_test)

from keras.layers import LSTM
model=Sequential()
model.add(Embedding(max_features,100,input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="rmsprop",
             loss="binary_crossentropy",
             metrics=["acc"])

model.summary()

model.fit(train_data,y_train,
         epochs=10,
         batch_size=32,
         validation_split=0.2)

model.evaluate(test_data,y_test)



