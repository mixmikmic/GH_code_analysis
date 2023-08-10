# we initialize some hyperparams
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
INDEX_FROM=3
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS, index_from=INDEX_FROM)

x_train[0][:5]

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[0] ))

from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

y_train

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding

embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

import csv

with open('word_metadata.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
       
    for key,value in sorted(id_to_word.items()):
        writer.writerow([value.encode('utf8')])

from keras.models import Model
from keras.callbacks import TensorBoard

model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

embedding_metadata = {
    embedding_layer.name: 'word_metadata.csv'
}

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_split=VALIDATION_SPLIT,
          callbacks=[TensorBoard(log_dir='word_reps', embeddings_freq=1, embeddings_metadata=embedding_metadata)])

