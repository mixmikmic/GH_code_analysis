# For a broad introduction to the problem and dataset: https://arxiv.org/pdf/1701.02145.pdf
# For modern results using deep learning: http://ieeexplore.ieee.org/document/7777224/

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# For the original '99 KDD dataset: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
# For the NSL-KDD Train+/Test+ data: https://github.com/defcom17/NSL_KDD

with open('kddcup.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]

# The Train+/Test+ datasets include sample difficulty rating and the attack class

kdd_cols += ['class', 'difficulty']

kdd = pd.read_csv('KDDTrain+.txt', names=kdd_cols)
kdd_t = pd.read_csv('KDDTest+.txt', names=kdd_cols)

kdd.head()

# Consult the linked references for attack categories: 
# https://www.researchgate.net/post/What_are_the_attack_types_in_the_NSL-KDD_TEST_set_For_example_processtable_is_a_attack_type_in_test_set_Im_wondering_is_it_prob_DoS_R2L_U2R
# The traffic can be grouped into 5 categories: Normal, DOS, U2R, R2L, Probe
# or more coarsely into Normal vs Anomalous for the binary classification task

kdd_cols = [kdd.columns[0]] + sorted(list(set(kdd.protocol_type.values))) + sorted(list(set(kdd.service.values))) + sorted(list(set(kdd.flag.values))) + kdd.columns[4:].tolist()

attack_map = [x.strip().split() for x in open('training_attack_types', 'r')]
attack_map = {k:v for (k,v) in attack_map}

attack_map

# Here we opt for the 5-class problem

kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)

def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)

def log_trns(df, col):
    return df[col].apply(np.log1p)

cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = cat_encode(kdd, col)
    kdd_t = cat_encode(kdd_t, col)

log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    kdd[col] = log_trns(kdd, col)
    kdd_t[col] = log_trns(kdd_t, col)

kdd = kdd[kdd_cols]
for col in kdd_cols:
    if col not in kdd_t.columns:
        kdd_t[col] = 0
kdd_t = kdd_t[kdd_cols]

# Now we have used one-hot encoding and log scaling

kdd.head()

difficulty = kdd.pop('difficulty')
target = kdd.pop('class')
y_diff = kdd_t.pop('difficulty')
y_test = kdd_t.pop('class')

target = pd.get_dummies(target)
y_test = pd.get_dummies(y_test)

target

y_test

target = target.values
train = kdd.values
test = kdd_t.values
y_test = y_test.values

# We rescale features to [0, 1]

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

train.shape

for idx, col in enumerate(list(kdd.columns)):
    print(idx, col)

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.embeddings import Embedding

# We apply a fairly simple MLP architecture

def build_network():

    models = []
    model = Sequential()
    model.add(Dense(64, input_dim=122))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# We use early stopping on a holdout validation set

NN = build_network()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])

from sklearn.metrics import confusion_matrix
preds = NN.predict(test)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

NN.evaluate(test, y_test)

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model

confusion_matrix(true_lbls, pred_lbls)

from sklearn.metrics import f1_score
f1_score(true_lbls, pred_lbls, average='weighted')

# Overall, we report similar model performance to the reference above.
# Their research suggest using unsupervised pretraining with autoencoders over
# both train and test before adding classifier layers for fine-tuning.
# I have done no parameter tuning but report comparable performance.
# Note the model has diffuculty with U2R and R2L.

