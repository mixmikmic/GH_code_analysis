import os
from Bio import SeqIO
import numpy as np
import tensorflow as tf
import timeit

base_pairs = {'A': [1, 0, 0, 0], 
'C': [0, 1, 0, 0],
'G': [0, 0, 1, 0],
'T': [0, 0, 0, 1],
'a': [1, 0, 0, 0],
'c': [0, 1, 0, 0],
'g': [0, 0, 1, 0],
't': [0, 0, 0, 1],
'n': [0, 0, 0, 0],
'N': [0, 0, 0, 0]}

def align_sequence(length, sequence):
    if len(sequence) > length:
        aligned_seq = sequence[:length]
    else:
        aligned_seq = sequence + [0]*(length-len(sequence))
    return aligned_seq

# NOT used
def to_positive_strand(strand, sequence):
    if strand == '-':
        unflattened_seq = [base_pairs[n] for n in sequence.complement()]
    else:
        unflattened_seq = [base_pairs[n] for n in sequence]

def process_seq_record(seq_record, X, y, length_read):
    header = seq_record.description.split('|')
    expressed = int(header[1])
    y.append(expressed)
    # unflattened_seq = to_positive_strand(header[3], seq_record.seq)
    # NO NEED to reverse complement
    unflattened_seq = [base_pairs[n] for n in seq_record.seq]
    flattened_seq = [i for x in unflattened_seq for i in x]
    aligned_seq = align_sequence(length_read, flattened_seq)
    X.append(np.array(aligned_seq))

def read_file(file, X, y, length_read):
    seq_record_list = list(SeqIO.parse("../data/input/3.24_species_only/" + file,"fasta"))
    for i in range(len(seq_record_list)):
        process_seq_record(seq_record_list[i], X, y, length_read)

def prepare_input(training_size, test_size, length_read):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    file_count = 0
    for file in os.listdir("../data/input/3.24_species_only"):
        if file.endswith(".fa"):
            if (file_count < training_size):
                read_file(file, X_train, y_train, length_read)
            elif (file_count < training_size + test_size):
                read_file(file, X_test, y_test, length_read)
            file_count += 1
    return X_train, y_train, X_test, y_test

def to_np_array(X_train, y_train, X_test, y_test):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if len(y_train.shape) == 1:
        y_train = np.transpose(np.array([y_train]))
    X_test = np.array(X_test)
    y_test = np.transpose(np.array(y_test))
    if len(y_test.shape) == 1:
        y_test = np.transpose(np.array([y_test]))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_input(200, 20, 4000)
X_train, y_train, X_test, y_test = to_np_array(X_train, y_train, X_test, y_test)
[X_train.shape, y_train.shape, X_test.shape, y_test.shape]

from sklearn import linear_model as lm

model = lm.LogisticRegression()
model.fit(X_train, y_train.ravel())

y_predicted = np.array(model.predict(X_test))
round(sum(y_test.ravel() == y_predicted)/y_test.shape[0], 3)

from keras.layers import Input, Dense
from keras.models import Model, Sequential

def train_nn(X_train, y_train, pr):
    model = Sequential()
    model.add(Dense(units=1000, activation='relu', input_dim=4000))
    model.add(Dense(units=400, activation='relu'))
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=10, activation='elu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer = 'SGD',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    model.fit(X_train, y_train, batch_size=100, epochs=5, verbose = pr)
    return model

def test_accuracy(model, X_test, y_test):
    result = model.predict(X_test)
    correct = list(np.apply_along_axis(lambda x: 0 if x<0.5 else 1, 1, result))==y_test.ravel()
    return round(sum(correct)/y_test.shape[0], 3)

test_accuracy(train_nn(X_train, y_train, 1), X_test, y_test)

file_num = list(range(100, 3000, 100))
read_time_taken = []

for training_size in range(100, 1000, 100):
    test_size = training_size * 0.1
    start_time = timeit.default_timer()
    X_train, y_train, X_test, y_test = prepare_input(training_size, test_size, 4000)
    X_train, y_train, X_test, y_test = to_np_array(X_train, y_train, X_test, y_test)
    elapsed = timeit.default_timer() - start_time
    print("Time to read " + str(int(training_size * 1.1)) + " files is " + str(round(elapsed, 2)) + " seconds")
    time_taken.append(elapsed)

file_num = list(range(100, 3000, 100))
total_time_taken = []

for training_size in range(100, 1000, 100):
    test_size = training_size * 0.1
    start_time = timeit.default_timer()
    X_train, y_train, X_test, y_test = prepare_input(training_size, test_size, 4000)
    X_train, y_train, X_test, y_test = to_np_array(X_train, y_train, X_test, y_test)
    result = test_accuracy(train_nn(X_train, y_train, 0), X_test, y_test)
    elapsed = timeit.default_timer() - start_time
    print("Time to read and train with " + str(int(training_size * 1.1)) + " files is " + str(round(elapsed, 2)) + " seconds"
         + ", the result accuracy is " + str(result))
    total_time_taken.append(elapsed)

import matplotlib.pyplot as plt
plt.plot(list(range(100, 1000, 100)), total_time_taken)

