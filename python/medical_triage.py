from os import listdir
from os.path import isfile, join
import pickle
import re
import random
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.callbacks import TensorBoard

def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

import pandas as pd
df=pd.read_csv("./data/phrases_embed.csv")
df = df[["Disease", "class"]]
df.head(3)

documents=df.as_matrix(columns=df.columns[0:1])
documents = documents.reshape(documents.shape[0])
print("documents.shape: {}".format(documents.shape))
body_positions=df.as_matrix(columns=df.columns[1:])
body_positions = body_positions.reshape(body_positions.shape[0])
print("body_positions.shape: {}".format(body_positions.shape))

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanUpSentence(r, stop_words = None):
    r = r.lower().replace("<br />", " ")
    r = re.sub(strip_special_chars, "", r.lower())
    if stop_words is not None:
        words = word_tokenize(r)
        filtered_sentence = []
        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
        return " ".join(filtered_sentence)
    else:
        return r

totalX = []
totalY = []
stop_words = set(stopwords.words("english"))
for i, doc in enumerate(documents):
    totalX.append(cleanUpSentence(doc, stop_words))
    body_positions[i] = re.sub(strip_special_chars, "", body_positions[i].lower())
    totalY.append(body_positions[i])

xLengths = [len(word_tokenize(x)) for x in totalX]
h = sorted(xLengths)  #sorted lengths
maxLength =h[len(h)-1]
print("max input length is: ",maxLength)

max_vocab_size = 30000
input_tokenizer = Tokenizer(max_vocab_size)
input_tokenizer.fit_on_texts(totalX)
input_vocab_size = len(input_tokenizer.word_index) + 1
print("input_vocab_size:",input_vocab_size)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))

totalX[1]

__pickleStuff("./data/input_tokenizer.p",input_tokenizer)

totalY[0:3]

target_tokenizer = Tokenizer(30)
target_tokenizer.fit_on_texts(totalY)
target_vocab_size = len(target_tokenizer.word_index) + 1
totalY = np.array(target_tokenizer.texts_to_sequences(totalY)) -1
totalY = totalY.reshape(totalY.shape[0])

print("target_vocab_size:",target_vocab_size)

totalY[0:3]

totalY = to_categorical(totalY, num_classes=target_vocab_size) # turn output to one-hot vecotrs

totalY[0:3]

vocab_size = input_vocab_size # vocab_size for model word embeding input
output_dimen = totalY.shape[1] # number of unique output classes

target_reverse_word_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
target_reverse_word_index[2]

metaData = {"maxLength":maxLength,"vocab_size":vocab_size,"output_dimen":output_dimen,"target_reverse_word_index":target_reverse_word_index}
__pickleStuff("./data/metaData_triage.p", metaData)

embedding_dim = 256
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length = maxLength))
# Each input would have a size of (maxLengthx256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
# All the intermediate outputs are collected and then passed on to the second GRU layer.
model.add(GRU(256, dropout=0.9, return_sequences=True))
# Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
model.add(GRU(256, dropout=0.9))
# The output is then sent to a fully connected layer that would give us our final output_dim classes
model.add(Dense(output_dimen, activation='softmax'))
# We use the adam optimizer instead of standard SGD since it converges much faster
tbCallBack = TensorBoard(log_dir='./Graph/medical_triage', histogram_freq=0,
                            write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(totalX, totalY, validation_split=0.1, batch_size=32, epochs=40, verbose=1, callbacks=[tbCallBack])
model.save('./data/triage.HDF5')

print("Saved model!")

model = None
target_reverse_word_index = None
maxLength = 0
def loadModel():
    global model, target_reverse_word_index, maxLength
    metaData = __loadStuff("./data/metaData_triage.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    target_reverse_word_index = metaData.get("target_reverse_word_index")
    embedding_dim = 256
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(GRU(256, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        model.add(Dense(output_dimen, activation='softmax'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('./data/triage.HDF5')
        model.summary()
    print("Model weights loaded!")

def findFeatures(text):
    textArray = [text]
    input_tokenizer = __loadStuff("./data/input_tokenizer.p")
    textArray = np.array(pad_sequences(input_tokenizer.texts_to_sequences(textArray), maxlen=maxLength))
    return textArray
def predictResult(text):
    global model, target_reverse_word_index
    if model is None:
        print("Please run \"loadModel\" first.")
        return None
    features = findFeatures(text)
    predicted = model.predict(features)[0]
    predicted = np.array(predicted)
    probab = predicted.max()
    predition = target_reverse_word_index[predicted.argmax()+1]
    return predition, probab

loadModel()

predictResult("Skin is quite itchy.")

predictResult("Sore throat fever fatigue.")

predictResult("Lower back hurt, so painful.")

predictResult("Very painful with period.")

predictResult("Sudden abdominal pain.")



