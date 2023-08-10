get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('aimport', 'helper, tests')
get_ipython().run_line_magic('autoreload', '1')

import collections

import helper
import numpy as np
import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')

for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # Keras Tokenizer Instance
    t = Tokenizer()
    
    # Fit the tokenizer on the list of sentences/strings
    t.fit_on_texts(x)
    
    # Tokenize the data
    tokenized_text = t.texts_to_sequences(x)
    
    return tokenized_text, t
tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    
    padded_sequences = pad_sequences(x,                 # List of sequences 
                                     maxlen= length,    # Length to sequnce the pad
                                     padding= 'post')   # Pad after each sequence
    
    return padded_sequences
tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')

from keras.models import Sequential
from keras.layers import Dense, Activation,SimpleRNN, Dropout, GRU, TimeDistributed

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Build the layers
    model = Sequential()
    
#     model.add(SimpleRNN(output_sequence_length,  # Dimensionality of the output space
#                        activation= 'tanh',       # tanh activation
#                        return_sequences = True,  # To feed the output to other recurrent layers
#                        input_shape= input_shape[1:]  # Input Dimensionality
#                        )
#              )
    model.add(GRU(128, input_shape= input_shape[1:], return_sequences= True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(french_vocab_size*5, activation= 'relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(french_vocab_size, activation= 'softmax')))
    
    learning_rate = 0.001
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_simple_model(simple_model)

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
print()
print("Original Text: ")
print(english_sentences[:1])
print("Original Transaltion: ")
print(french_sentences[:1])

# Evaluate Model
score = simple_rnn_model.evaluate(tmp_x, preproc_french_sentences, verbose=0)
print("Train accurancy: ", score[1])

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    model = Sequential()
    # Embedding Layer
    model.add(Embedding(english_vocab_size,                # Vocab Size
                        128,                               # Embedding Dimension 
                        input_length= input_shape[1],      # Length of Input 
                        input_shape= input_shape[1:]))     # Shape of Input
    # GRU Layer
    model.add(GRU(128, return_sequences= True))
    # Dense Connected Layer
    model.add(TimeDistributed(Dense(french_vocab_size * 5, activation= 'relu')))
    model.add(Dropout(0.2))
    # Output Layer
    model.add(TimeDistributed(Dense(french_vocab_size, activation= 'softmax')))
    # Compile Model
    learning_rate = 0.001
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_embed_model(embed_model)


# Reshape the input
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)

# Train the neural network
embed_rnn_model= embed_model(tmp_x.shape,
                    max_french_sequence_length,
                    english_vocab_size,
                    french_vocab_size)
embed_rnn_model.summary()
embed_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs= 10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
print()
print("Original Text: ")
print(english_sentences[:1])
print("Original Transaltion: ")
print(french_sentences[:1])

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    model = Sequential()
    # Bidirectional Layer
    model.add(Bidirectional(GRU(128, return_sequences= True), input_shape= input_shape[1:]))
    # Dense Connected Layer
    model.add(TimeDistributed(Dense(french_vocab_size * 5, activation= 'relu')))
    model.add(Dropout(0.2))
    # Output Layer
    model.add(TimeDistributed(Dense(french_vocab_size, activation= 'softmax')))
    # Compile Model
    learning_rate = 0.001
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_bd_model(bd_model)

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train and Print prediction(s)
bd_rnn_model= bd_model(tmp_x.shape,
                    max_french_sequence_length,
                    english_vocab_size,
                    french_vocab_size)
bd_rnn_model.summary()
bd_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs= 10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(bd_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
print()
print("Original Text: ")
print(english_sentences[:1])
print("Original Transaltion: ")
print(french_sentences[:1])

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    model = Sequential()
    
    # Encoder
    model.add(GRU(128, input_shape= input_shape[1:]))
    model.add(RepeatVector(output_sequence_length))
    
    # Decoder
    model.add(GRU(128, return_sequences= True))
    model.add(TimeDistributed(Dense(french_vocab_size * 5, activation= 'relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(french_vocab_size, activation= 'softmax')))
    # Compile Model
    learning_rate = 0.001
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_encdec_model(encdec_model)


tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train and Print prediction(s)
encdec_rnn_model= encdec_model(tmp_x.shape,
                    max_french_sequence_length,
                    english_vocab_size,
                    french_vocab_size)
encdec_rnn_model.summary()
encdec_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs= 10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(encdec_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
print()
print("Original Text: ")
print(english_sentences[:1])
print("Original Transaltion: ")
print(french_sentences[:1])

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    model = Sequential()
    # 1. Embedding Layer
    model.add(Embedding(english_vocab_size, 128,
                        input_length= input_shape[1], 
                        input_shape= input_shape[1:]))
    # 2. Bidirectional Layer
    model.add(Bidirectional(GRU(512)))
    # 3. Encoder
    model.add(RepeatVector(output_sequence_length))
    # 4. Decoder
    model.add(Bidirectional(GRU(512, return_sequences= True)))
    model.add(Dropout(0.2))
    # 5. Fully Connected Layer
    model.add(TimeDistributed(Dense(french_vocab_size * 5, activation='relu')))
    model.add(Dropout(0.2))
    # 6. Output Layer
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    # Compile Model
    learning_rate = 0.003
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model
tests.test_model_final(model_final)


print('Final Model Loaded')

def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # Train neural network using model_final
    model = model_final(x.shape, 
                        y.shape[1], 
                        len(x_tk.word_index) + 1, 
                        len(y_tk.word_index) + 1)
    model.summary()
    model.fit(x, y, batch_size=1024, epochs= 10, validation_split=0.2)

    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)

get_ipython().getoutput('python -m nbconvert *.ipynb')

