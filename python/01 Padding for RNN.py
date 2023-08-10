import numpy as np
from keras.preprocessing.sequence import pad_sequences

data = np.array([[1,2,3], [4,5,6]])
pad_sequences(data, maxlen=10)

data = np.array([[1,2,3], [4,5,6]])
pad_sequences(data, maxlen=10, padding='post')

data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                 [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]])
pad_sequences(data, maxlen=3, truncating='pre')

data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                 [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]])
pad_sequences(data, maxlen=3, truncating='post')

