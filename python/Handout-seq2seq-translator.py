from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU
from keras.callbacks import ModelCheckpoint

import numpy as np

from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

data_path = 'spa-eng/spa.txt' # Path to the data txt file on disk.
num_samples = 100000  # Number of samples to train on.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path).read().split('\n')
#for line in lines[: min(num_samples, len(lines) - 1)]:
for line in lines[:num_samples]:
    input_text, target_text = line.split('\t')
    if len(target_text) > 100 or len(input_text) > 100:
        break
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

for i in range(0,10000, 1000):
    print(input_texts[i],target_texts[i])

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

print("encoder_input_data shape:", encoder_input_data.shape)
print("decoder_input_data shape:", decoder_input_data.shape)

latent_dim = 256  # Latent dimensionality of the encoding space.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

batch_size = 64  # Batch size for training.
epochs = 30  # Number of epochs to train for.

# Run training
callback = ModelCheckpoint('s2s_weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                           save_best_only=True, save_weights_only=True)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05,
          callbacks=[callback])

# This cell load a pretrained model that was trained with 100000 samples
# The values of the following variables have to be set before defining the model:
# Number of unique input tokens: 86
# Number of unique output tokens: 106
# Max sequence length for inputs: 46
# Max sequence length for outputs: 85

model.load_weights('s2s_weights.21-0.55.hdf5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
encoder_model.summary()
decoder_model.summary()

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def one_hot_encoder_input(input_text):
    result = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')    
    for t, char in enumerate(input_text):
        result[0, t, input_token_index[char]] = 1.
    return result

decode_sequence(one_hot_encoder_input("I didn't know you."))

def beam_search(states_value, target_idx, lahead=3, 
                nbeams = 2, min_nbeams = 1, decrement = 0):
    nbeams = max(nbeams, min_nbeams)
    if reverse_target_char_index[target_idx] == '\n' or lahead < 1:
        return ([], 0, 0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_idx] = 1.
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + states_value)
    states_value = [h, c]
    # Sample a token
    candidates = [(output_tokens[0, -1, i], i) for i in range(output_tokens.shape[2])]
    candidates.sort()
    best_prob = float('-inf')
    best_idx = 0
    for prob, idx in candidates[-nbeams:]:
        _, _, cand_log_prob = beam_search(states_value, idx, lahead - 1,
                                                           nbeams - decrement, min_nbeams, decrement)
        cand_log_prob += np.log(prob)
        if cand_log_prob > best_prob:
            best_prob = cand_log_prob
            best_idx = idx
    return (states_value, best_idx, best_prob)

def decode_sequence_beam(input_seq, lahead=3, nbeams = 2, min_nbeams = 1, decrement = 0):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    decoded_sentence = ''
    best_idx = target_token_index['\t']
    for i in range(max_decoder_seq_length):
        states_value, best_idx, best_prob = beam_search(states_value, best_idx , 
                                                        nbeams = nbeams, 
                                                        min_nbeams = min_nbeams, 
                                                        decrement = decrement)
        sampled_char = reverse_target_char_index[best_idx]
        if sampled_char == '\n':
            break
        decoded_sentence += sampled_char
    return decoded_sentence

decode_sequence_beam(one_hot_encoder_input("I didn't know you."), lahead=4, nbeams=4)



