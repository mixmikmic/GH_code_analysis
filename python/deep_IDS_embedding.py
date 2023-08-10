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

len(np.unique(kdd['flag'].values))

# Consult the linked references for attack categories: 
# https://www.researchgate.net/post/What_are_the_attack_types_in_the_NSL-KDD_TEST_set_For_example_processtable_is_a_attack_type_in_test_set_Im_wondering_is_it_prob_DoS_R2L_U2R
# The traffic can be grouped into 5 categories: Normal, DOS, U2R, R2L, Probe
# or more coarsely into Normal vs Anomalous for the binary classification task

kdd_cols = kdd.columns.tolist()
kdd_cols.remove('protocol_type')
kdd_cols.remove('service')
kdd_cols.remove('flag')
kdd_cols += ['protocol_type', 'service', 'flag']

attack_map = [x.strip().split() for x in open('training_attack_types', 'r')]
attack_map = {k:v for (k,v) in attack_map}

attack_map

# Here we opt for the 5-class problem

kdd['class'] = kdd['class'].replace(attack_map)
kdd_t['class'] = kdd_t['class'].replace(attack_map)

def ent_encode(df, col):
    vals = sorted(np.unique(df[col].values))
    val_dict = {val:idx for idx, val in enumerate(vals)}
    df[col] = df[col].map(val_dict)
    return df

def log_trns(df, col):
    return df[col].apply(np.log1p)

cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    kdd = ent_encode(kdd, col)
    kdd_t = ent_encode(kdd_t, col)

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
cat_mat = train[:,-3:]
train = train[:,:-3]
cat_tst = test[:,-3:]
test = test[:,:-3]

# We rescale features to [0, 1]

min_max_scaler = MinMaxScaler()
train = min_max_scaler.fit_transform(train)
test = min_max_scaler.transform(test)

for idx, col in enumerate(list(kdd.columns)):
    print(idx, col)

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.embeddings import Embedding

# We apply entity embedding for the label encoded features
# The input must be a list of arrays for each categorical
# feature as well as the array of continuous normalized features

train = [train] + [col for col in cat_mat.T]
test = [test] + [col for col in cat_tst.T]

def build_network():

    models = []
    
    model_dens = Sequential()
    model_dens.add(Dense(36, input_dim=38))
    model_dens.add(Activation('relu'))
    model_dens.add(Dropout(.15))
    model_dens.add(Dense(16))
    models.append(model_dens)

    model_proto = Sequential()
    model_proto.add(Embedding(3, 2, input_length=1))
    model_proto.add(Reshape(target_shape=(2,)))
    models.append(model_proto)

    model_serv = Sequential()
    model_serv.add(Embedding(70, 4, input_length=1))
    model_serv.add(Reshape(target_shape=(4,)))
    models.append(model_serv)

    model_flag = Sequential()
    model_flag.add(Embedding(11, 3, input_length=1))
    model_flag.add(Reshape(target_shape=(3,)))
    models.append(model_flag)
    
    model = Sequential()
    model.add(Merge(models, mode='concat'))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# We use early stopping on a holdout validation set

NN = build_network()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=32, callbacks=[early_stopping])

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

# This model also performs similarly though slightly worse.
# Note that this model shows less bias to classify as 'normal'
# This architecture may perform well in binary classification.

