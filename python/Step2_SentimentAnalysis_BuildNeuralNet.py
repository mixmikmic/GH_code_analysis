import numpy as np
import pandas as pd
import pickle

from news_text_clean import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

# Fix random seed for reproducibility
np.random.seed(7) 
from tensorflow import set_random_seed
set_random_seed(2)

# Load the data
data1 = pd.read_json('data/manualVerified_senti_2600.json')

# Shuffle the data
data = data1.sample(frac=1,random_state=11).reset_index(drop=True)

# Clean the data
data = news_text_clean(data)

# USER INPUT
top_words = 10000 # Keep only the top 10000 frequently occuring words
max_words = 500 # Max words to consider in a given article
embed_dim = 32

# Preprocess the text (Convert words to numbers)
tokeniz = Tokenizer(num_words=top_words, split=' ')
tokeniz.fit_on_texts(data['contents'].values)

word_index = tokeniz.word_index
print('Total numbers of words in the dataset = ', len(word_index))

X_data = tokeniz.texts_to_sequences(data['contents'].values)
X_data = pad_sequences(X_data, maxlen=max_words, truncating='post')

# Negative, neutral & positive sentiments are -1, 0 & 1 in data. Convert it to 0, 1 & 2
y_data = data['Sentiment']+1 

# Split the data into train (90% of the data) and test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.9, random_state=1)

print('Counts of Negative, Neutral and Positive article =',np.unique(y_train,return_counts=True))

Y_train = np_utils.to_categorical(y_train, 3)
Y_test = np_utils.to_categorical(y_test, 3)

# Build the CNN model
model = Sequential()

model.add(Embedding(top_words, embed_dim, input_length=max_words))

model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# Apply the model on the test set data
Y_prob = model.predict(X_test)
Y_class = np.argmax(Y_prob, axis=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy of the CNN is", scores[1]*100, "%")

# Calculate confusion matrix from sklearn
from sklearn.metrics import confusion_matrix
confu_mat = confusion_matrix(y_test, Y_class)

print(confu_mat)

# Create a Pandas dataframe for Seaborn
confu_df = pd.DataFrame(confu_mat, columns=['Pred_0','Pred_1','Pred_2'], index=['True_0','True_1','True_2'])

plt.figure
import seaborn as sn
sn.heatmap(confu_df, annot=True)
plt.show()

# Save the model and tokenizer
model.save('model_content_1.h5')
pickle.dump(tokeniz, open('tokenizer_content_1.p','wb'))

