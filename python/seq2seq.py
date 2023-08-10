from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 50000
DIGITS = 3
INVERT = False
# Try replacing GRU, or SimpleRNN
RNN = recurrent.GRU
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)

questions = []
expected = []
seen = set()

get_ipython().run_cell_magic('time', '', "\n#Generating random  numbers to perofrm addition on\nprint('Generating data...')\nwhile len(questions) < TRAINING_SIZE:\n    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))\n    a, b = f(), f()\n    # Skip any addition questions we've already seen\n    # Also skip any such that X+Y == Y+X (hence the sorting)\n    key = tuple(sorted((a, b)))\n    if key in seen:\n        continue\n    seen.add(key)\n    # Pad the data with spaces such that it is always MAXLEN\n    q = '{}+{}'.format(a, b)\n    query = q + ' ' * (MAXLEN - len(q))\n    ans = str(a + b)\n    # Answers can be of maximum size DIGITS + 1\n    ans += ' ' * (DIGITS + 1 - len(ans))\n    if INVERT:\n        query = query[::-1]\n    questions.append(query)\n    expected.append(ans)\nprint('Total addition questions:', len(questions))\n\n#We now have 50000 examples of addition, each exaple contains the addition between two numbers\n#Each example contains the first number followed by '+' operand followed by the second number \n#examples - 85+96, 353+551, 6+936\n#The answers to the additon operation are stored in expected")

print(len(questions), len(expected))
print(questions[0], expected[0])

get_ipython().run_cell_magic('time', '', "\n#The above questions and answers are going to be one hot encoded, \n#before training.\n#The encoded values will be used to train the model\n#The maximum length of a question can be 7 \n#(3 digits followed by '+' followed by 3 digits)\n#The maximum length of an answer can be 4 \n#(Since the addition of 3 digits yields either a 3 digit number or a 4\n#4 digit number)\n\n#Now for training each number or operand is going to be one hot encode below\n#In one hot encode there are 12 possibilities '0123456789+ ' (The last one is a space)\n#Since we assume a maximum of 3 digit numbers, a two digit number is taken as space with two digts, or \n#a single digit number as two spaces with a number\n\n#So for questions we get 7 rows since the max possible length is 7, and each row has a length of 12 because it will\n#be one hot encoded with True and False, depending on the character(any one of the number, '+' operand, or space)\n#will be stored  in X_train and X_val\n#The 4th position in(1,2,3,4,5,6,7) will indicate the one hot encoding of the '+' operand\n\n##So for questions we get 4 rows since the max possible length is 4, and each row has a length of 12 because it will\n#be one hot encoded with True and False, depending on the character(any one of the number, '+' operand, or space)\n#will be stored  in y_train and y_val\n\n\nprint('Vectorization...')\nX = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)\ny = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)\nfor i, sentence in enumerate(questions):\n    X[i] = ctable.encode(sentence, maxlen=MAXLEN)\nfor i, sentence in enumerate(expected):\n    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)\n\n# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits\nindices = np.arange(len(y))\nnp.random.shuffle(indices)\nX = X[indices]\ny = y[indices]\n\n# Explicitly set apart 10% for validation data that we never train over\nsplit_at = len(X) - len(X) / 10\n(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))\n(y_train, y_val) = (y[:split_at], y[split_at:])")

print(X.shape, y.shape)

print(X[0])
print
print(y[0])

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

get_ipython().run_cell_magic('time', '', '\n#Training the model with the encoded inputs\nprint(\'Build model...\')\nmodel = Sequential()\n# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE\n# note: in a situation where your input sequences have a variable length,\n# use input_shape=(None, nb_feature).\nmodel.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))\n# For the decoder\'s input, we repeat the encoded input for each time step\nmodel.add(RepeatVector(DIGITS + 1))\n# The decoder RNN could be multiple layers stacked or a single layer\nfor _ in range(LAYERS):\n    model.add(RNN(HIDDEN_SIZE, return_sequences=True))\n\n# For each of step of the output sequence, decide which character should be chosen\nmodel.add(TimeDistributed(Dense(len(chars))))\nmodel.add(Activation(\'softmax\'))')

from keras.callbacks import EarlyStopping, ModelCheckpoint, RemoteMonitor

remote = RemoteMonitor(root='http://localhost:9000')

# compile and run
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

result = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          nb_epoch=124,
          validation_data=(X_val, y_val),
          verbose=2,
          callbacks = [
                         EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
                         ModelCheckpoint('{}-progress-dl'.format("learn-to-execute"), monitor='val_loss', verbose=True, save_best_only=True), remote],
         )

nb_end_epoch = len(result.history['acc'])
print("End at {} epoch".format(nb_end_epoch))

TARGET_NAME = "learn-to-execute"

from sklearn import metrics
from keras.utils import np_utils

# -- load in best network
model.load_weights('{}-progress-dl'.format(TARGET_NAME))
score = model.evaluate(X_val, y_val, verbose=0)
print('\n')
print('Test score:', score[0])
print('Test accuracy:', score[1])

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
get_ipython().magic('matplotlib inline')
plt.style.use("ggplot")

from IPython.display import SVG, display
from keras.utils.visualize_util import model_to_dot, plot

x = range(nb_end_epoch)
fix, ax = plt.subplots(figsize=(10, 4))
plt.suptitle('Loss and Accuracy GRU')
plt.subplot(1, 2, 1)
plt.plot(x, result.history['acc'], label="train acc")
plt.plot(x, result.history['val_acc'], label="val acc")
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.title("Accuracy plot");
plt.subplot(1, 2, 2)
plt.plot(x, result.history['loss'], label="train loss")
plt.plot(x, result.history['val_loss'], label="val loss")
plt.legend(loc='best', bbox_to_anchor=(1,0.5))
plt.title("Loss plot");
plt.show()

import h5py
import json
from keras.models import model_from_json, load_model

model.summary()

model.save("./{}.h5".format(TARGET_NAME))

get_ipython().run_cell_magic('time', '', "\n#For predicting the outputs, the predict method will return \n#an one hot encoded ouput, we decode the one hot encoded \n#ouptut to get our final output\n\n# Select 10 samples from the validation set at random so we can visualize errors\nfor i in range(10):\n    ind = np.random.randint(0, len(X_val))\n    rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]\n    preds = model.predict_classes(rowX, verbose=0)\n    q = ctable.decode(rowX[0])\n    correct = ctable.decode(rowy[0])\n    guess = ctable.decode(preds[0], calc_argmax=False)\n    print('Q', q[::-1] if INVERT else q)\n    print('T', correct)\n    print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)\n    print('---')")

# kindergarten
q='1+1'
a='2'

# primary school
q='23+75'
a='98'

# high school
q='127+249'
a='376'

# university
q='123+897'
a='1020'

xq = ctable.encode("{:7s}".format(q), maxlen=MAXLEN)
xa = ctable.encode("{:4s}".format(a), maxlen=MAXLEN+1)

preds = model.predict_classes(np.asarray([xq]), verbose=0)
guess = ctable.decode(preds[0], calc_argmax=False)
print('Q', q)
print('T', a)
print(colors.ok + '☑' + colors.close if a.strip() == guess.strip() else colors.fail + '☒' + colors.close, guess)
print('---')

import re

def fix_math(q, model=model):
    xq = re.findall(r'\d+\+\d+', q)
    if xq:
        xq = xq[0]
        xq = ctable.encode("{:7s}".format(q), maxlen=MAXLEN)
        preds = model.predict_classes(np.asarray([xq]), verbose=0)
        guess = ctable.decode(preds[0], calc_argmax=False)
        return u"👻 \033[35mI guess: " + guess.strip() + "\033[0m"
    else:
        return u"👻 \033[91mGive me correct summation, LOL!!!\033[0m"

print(fix_math("12+23"))

print(fix_math(""))



