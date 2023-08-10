import os

#if using Theano with GPU
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import random
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

# load up our text
text_files = glob('../data/sotu/*.txt')
text_files[0:5]

# let us create a long string variable text
text = '\n'.join([open(f, 'r').read() for f in text_files])
text[0:500]

print(type(text))
print(len(text))

# extract all (unique) characters
# these are our "categories" or "labels". We want to predict the next character from the past few (e.g 20) characters
chars = list(set(text))
print(chars)

len(chars)

# set a fixed vector size
# so we look at specific windows of characters
max_len = 20

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.summary()

example_text = "The fish trap exists because of the fish. Once you have gotten the fish you can forget the trap. The rabbit snare exists because of the rabbit. Once you have gotten the rabbit, you can forget the snare. Words exist because of meaning. Once you have gotten the meaning, you can forget the words. Where can I find a man who has forgotten words so that I may have a word with him?"

# step size here is 3, but we can vary that
input_1 = example_text[0:20]
true_output_1 = example_text[20]
# >>> 'The fish trap exists'
# >>> ' '
print(input_1)
print(true_output_1)

input_2 = example_text[3:23]
true_output_2 = example_text[23]
# >>> 'fish trap exists be'
# >>> 'c'
print(input_2)
print(true_output_2)

input_3 = example_text[6:26]
true_output_3 = example_text[26]
# >>> 'sh trap exists becau'
# >>> 's'

# etc
print(input_3)
print(true_output_3)

step = 3
inputs = []
outputs = []
for i in range(0, len(text) - max_len, step):
    inputs.append(text[i:i+max_len])
    outputs.append(text[i+max_len])

print(len(inputs),len(outputs))

# Let us see a specific example for the input text and the output
print(inputs[1995], outputs[1995])

char_labels = {ch:i for i, ch in enumerate(chars)}
labels_char = {i:ch for i, ch in enumerate(chars)}

print(char_labels)

print(labels_char)

# assuming max_len = 7
# so our examples have 7 characters, e.g these ones:
example = 'cab dab'
# these are the character 'codes':
example_char_labels = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    ' ' : 4
}

# matrix form
# the example uses only five kinds of characters,
# so the vectors only need to have five components,
# and since the input phrase has seven characters,
# the matrix has seven vectors.
[
    [0, 0, 1, 0, 0], # c
    [1, 0, 0, 0, 0], # a
    [0, 1, 0, 0, 0], # b
    [0, 0, 0, 0, 1], # (space)
    [0, 0, 0, 1, 0], # d
    [1, 0, 0, 0, 0], # a
    [0, 1, 0, 0, 0]  # b
]

# using bool to reduce memory usage
X = np.zeros((len(inputs), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(inputs), len(chars)), dtype=np.bool)

print(X.shape)
print(y.shape)

# set the appropriate indices to 1 in each one-hot vector
for i, example in enumerate(inputs):
    for t, char in enumerate(example):
        X[i, t, char_labels[char]] = 1
    y[i, char_labels[outputs[i]]] = 1

#Let us plot a specific sentence
plt.imshow(X[1345,:,:])

# Let us look at an example input
print(X[13,15,:])

# Let us look at an example label
print(y[230,:])

# more epochs is usually better, but training can be very slow if not on a GPU
#epochs = 10
#model.fit(X, y, batch_size=128, epochs=epochs)

# Let us see how to sample from a Boltzmann distribution with parameters probs and temperature
probs=[3,1,1,4]
temperature=2.3

a = np.log(probs)/temperature
dist = np.exp(a)/np.sum(np.exp(a))
choices = range(len(probs))
print(dist)
print(choices)
print(np.random.choice(choices, p=dist))

# A function to draw samples from a Boltzmann distribution
def sample(probs, temperature):
    """samples an index from a vector of probabilities
    (this is not the most efficient way but is more robust)"""
    a = np.log(probs)/temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(probs))
    return np.random.choice(choices, p=dist)

temperature=0.35 
seed=None 
num_chars=100

#Let us define a lambda function that checks if a given sentence is a long enough
predict_more=lambda x: len(x) < num_chars
predict_more('This sentence is too short')

print(max_len)
# Let us select a random seed sentences
start_idx = random.randint(0, len(text) - max_len - 1)
seed = text[start_idx:start_idx + max_len]

sentence = seed
print(sentence)

# generate the input tensor
# from the last max_len characters generated so far
x = np.zeros((1, max_len, len(chars)))
for t, char in enumerate(sentence):
    x[0, t, char_labels[char]] = 1.
print(x.shape)

plt.imshow(x[0,:,:])

probs = model.predict(x, verbose=1)[0]

print(probs)
print(probs.shape)

# Based on these ideas, let us create a generate function
def generate(temperature=0.35, seed=None, num_chars=100):
    predict_more=lambda x: len(x) < num_chars
    
    if seed is not None and len(seed) < max_len:
        raise Exception('Seed text must be at least {} chars long'.format(max_len))

    # if no seed text is specified, randomly select a chunk of text
    else:
        start_idx = random.randint(0, len(text) - max_len - 1)
        seed = text[start_idx:start_idx + max_len]

    sentence = seed
    generated = sentence

    while predict_more(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_labels[char]] = 1.

        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_char = labels_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

# serialize model to JSON
model_json = model.to_json()
with open("text_gen_model.json", "w") as json_file:
    json_file.write(model_json)

# Let us train for 10 epochs
epochs = 10
for i in range(epochs):
    print('epoch %d'%i)

    # set nb_epoch to 1 since we're iterating manually
    # comment this out if you just want to generate text
    model.fit(X, y, batch_size=128, epochs=1)

    # preview
    for temp in [0.2, 0.5, 1., 1.2]:
        print('temperature: %0.2f'%temp)
        print('%s'%generate(temperature=temp))
    
    # serialize weights to HDF5
    fname="text_gen_model_params_"+str(i)+".h5"
    # save the model weights
    model.save_weights(fname)
    print("Saved model to disk")

# Let us generate some text
print('%s' % generate(temperature=0.4, seed='Today, we are facing an important challenge.', num_chars=2000))

# Let us load a saved model from disk
from keras.models import model_from_json
# load json and create model
json_file = open('text_gen_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
                 
model = model_from_json(loaded_model_json)
# load weights into new model
fname="text_gen_model_params_9.h5"
model.load_weights(fname)
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# generate text from the current model
print('%s' % generate(temperature=0.4, seed='Today, we are facing an important challenge.', num_chars=2000))



