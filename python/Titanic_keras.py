from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from sklearn import preprocessing

import numpy as np
import csv
import pandas as pd
import sys

np.random.seed(1919)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

y = train.Survived.values
train = train.drop(['Survived'], axis=1)


def modify_data(base_df):
    new_df = pd.DataFrame()
    new_df['Gender'] = base_df.Sex.map(lambda x:1 if x.lower() == 'female' else 0)
    # apply functions to dataframe => Fare NaN
    fares_by_class = base_df.groupby('Pclass').Fare.median()

    def getFare(example):
        if pd.isnull(example):
            example['Fare'] = fares_by_class[example['Pclass']]
        return example
    new_df['Fare'] = base_df['Fare']
    new_df['Family'] = (base_df.Parch + base_df.SibSp) > 0
    new_df['Family'] = new_df['Family'].map(lambda x:1 if x else 0)
    new_df['GenderFam'] = new_df['Gender']+new_df['Family']
    new_df['Title'] = base_df.Name.map(lambda x:x.split(' ')[0])
    new_df['Rich'] = base_df.Pclass == 1
    return new_df
    
train = modify_data(train)

# TEST DATA
#test = pd.read_csv('titanic_test.csv', header=0)        # Load the test file into a dataframe
ids = test['PassengerId'].values
test = modify_data(test)
train = train.fillna(-1)
test = test.fillna(-1)



for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

X = train.values
dimof_input = X.shape[1]
dimof_output = len(set(y.flat))
y = np_utils.to_categorical(y, dimof_output)
test_x = test.values

batch_size = 20
hidden_sizes = [200, 200]
dropout = 0.5
countof_epoch = 200
verbose = 0

model = Sequential()
for i, s in enumerate(hidden_sizes):
    if i:
        model.add(Dense(s))
    else:
        model.add(Dense(s, input_shape=(dimof_input,)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
model.add(Dense(dimof_output))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer="rmsprop")

model.fit(
    X, y,
    show_accuracy=True, #validation_split=0.2,
    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

# Evaluate
loss, accuracy = model.evaluate(X, y, show_accuracy=True, verbose=verbose)
print('loss: ', loss)
print('accuracy: ', accuracy)
print()
predict_x = test_x
predict_df = test
preds = model.predict(predict_x, batch_size=batch_size)
pred_arr = [p[0] for p in preds]
results = pd.DataFrame({"PassengerId":ids, 'Survived': pred_arr})
results['PassengerId'] = results['PassengerId'].astype('int')
results.Survived = results.Survived.map(lambda x:0 if x >= 0.5 else 1)
results.set_index("PassengerId")
print results.Survived.sum()
results.to_csv('results_nn.csv', index=False)



