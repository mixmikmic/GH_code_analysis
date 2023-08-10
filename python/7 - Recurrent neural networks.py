model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(256)))

# or

model.add(LSTM(256)) # Using default value for return_sequences, which is False
model.add(Dense(256))

x = Input(shape=(50, 100)) # 50 timesteps with 100 features each

h1_fw = GRU(256)(x)
h1_bw = GRU(256, go_backwards=True)(x)

h1 = Merge([h1_fw, h1_bw], mode='concat', concat_axis=-1)

import numpy as np
np.random.seed(42)  # for reproducibility

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, TimeDistributed
from keras.layers.recurrent import LSTM

# In case your RNN is backpropagating over a large number of timesteps, you might need to do this
# (only when using the Theano backend)
import sys
sys.setrecursionlimit(50000)

# Loading data. Input is already standardized (mean = 0, standard deviation = 1)
X = np.load('stft_data_noisy.npy') # shape is (n_samples, n_frames, n_frequency_bins)
y = np.load('stft_data_clean.npy') # same as above, but scaled to the range [0, 1]

X_train, X_test = X[:10000], X[10000:]
y_train, y_test = X[:10000], X[10000:]

model = Sequential()
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(y_train.shape[-1], activation='sigmoid')))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, batch_size=32, nb_epoch=100, validation_split=0.1)



