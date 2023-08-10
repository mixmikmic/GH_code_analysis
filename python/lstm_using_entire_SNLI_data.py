from __future__ import division, print_function, absolute_import
import json
from pprint import pprint
import pickle
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math

# setting the paths
filepath = '/home/pujun/Desktop/StanfordClasses/lstm for natural language understanding/transfer_learning_snli.jsonl'
percentage_split = .7

vocab = {}
word_count = 1

def parse_data(json_data):
    global word_count
    
    X = []
    Y = []
    for d in json_data:
        current_attribute_list = []
        words = tokenized_and_lowercase = word_tokenize(d['example'].lower())
        for w in words:
            if w not in vocab:
                vocab[w] = word_count
                word_count += 1
            current_attribute_list.append(vocab[w])
        X.append(current_attribute_list)
        Y.append(d['label'])

    return (X, Y)

data = []
with open(filepath) as f:
    for line in f:
        data.append(json.loads(line))
    X, Y = parse_data(data)

print("Number of examples:", len(X))
print("Number of distinct words:", word_count)


with open('SNLIdata','w') as f:
    pickle.dump(data,f)

data_length_list = [len(eg) for eg in X]
num_words_in_longest_sentence = max(data_length_list)

print("Length of the biggest sentence:", num_words_in_longest_sentence)

num_training_examples = int(math.ceil(len(X) * percentage_split))
print(num_training_examples)
trainX = X[:num_training_examples]
trainY = Y[:num_training_examples]

testX = X[num_training_examples:]
testY = Y[num_training_examples:]

# Data preprocessing
# Sequence padding 
trainX = pad_sequences(trainX, maxlen=num_words_in_longest_sentence, value=0.)
testX = pad_sequences(testX, maxlen=num_words_in_longest_sentence, value=0.)

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, num_words_in_longest_sentence])
net = tflearn.embedding(net, input_dim=word_count, output_dim=128)
net = tflearn.lstm(net, 128)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=128)



