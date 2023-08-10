import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from utils import mnist_reader
Labels = ["T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"]

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

N = X_train.shape[0]
batch_size = 32

n_batch = int(N/batch_size)

from keras.utils import to_categorical
y_train_ohe = to_categorical(y_train)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import keras.metrics as metrics

dropout_rate=0.1

model = Sequential()
model.add(Dense(units=200, input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(units=150))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(units=30))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(units=10))
model.add(Activation('softmax'))

def top3_acc(ytrue, ypred):
    return metrics.top_k_categorical_accuracy(ytrue, ypred, k=3)

# Change decay for better results

# lr: 1e-3, decay: 0

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001, decay=0., nesterov=False),
             metrics=[metrics.categorical_accuracy, top3_acc])

history = model.fit(X_train, y_train_ohe, epochs=5, batch_size=32)

plt.figure(figsize=(20,10))

for i, (name, values) in enumerate(history.history.items()):
    plt.subplot(1, len(history.history.items()), i+1)
    plt.plot(values)
    plt.title(name)

from utils.validation import validation_report
validation_report(test_data=X_test,
                 test_label=y_test,
                 model=model,
                 names=Labels)

