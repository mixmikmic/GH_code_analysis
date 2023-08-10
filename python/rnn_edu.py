import pandas as pd
import numpy as np
import json
from sklearn import model_selection
from sklearn import metrics 

filename = 'skill_builder_data_corrected.csv'
df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)
df = df[(df['original'] == 1) & (df['attempt_count'] == 1) & ~(df['skill_name'].isnull())]

def generate_datasets():
    users_list = df['user_id'].unique()
    skill_list = df['skill_name'].unique()
    
    skill_dict = dict(zip(skill_list, np.arange(len(skill_list), dtype='int32') + 1))
    response_list = []
    skill_list = []
    assistment_list = []
    
    counter = 0
    for user in users_list:
        sub_df = df[df['user_id'] == user]
        if len(sub_df) > 100:
            first_hundred = sub_df.iloc[0:100]
            response_df = pd.DataFrame(index=[counter], columns=['student_id']+['r'+str(i) for i in range(100)])
            skill_df = pd.DataFrame(index=[counter], columns=['student_id']+['s'+str(i) for i in range(100)])
            assistment_df = pd.DataFrame(index=[counter], columns=['student_id']+['a'+str(i) for i in range(100)])
            
            response_df.iloc[0, 0] = first_hundred.iloc[0]['user_id']
            skill_df.iloc[0, 0] = first_hundred.iloc[0]['user_id']
            assistment_df.iloc[0, 0] = first_hundred.iloc[0]['user_id']
            for i in range(100):
                response_df.iloc[0, i+1] = first_hundred.iloc[i]['correct']
                skill_df.iloc[0, i+1] = skill_dict[first_hundred.iloc[i]['skill_name']]
                assistment_df.iloc[0, i+1] = first_hundred.iloc[i]['assistment_id']
            counter += 1
            response_list.append(response_df)
            skill_list.append(skill_df)
            assistment_list.append(assistment_df)
    
    response_df = pd.concat(response_list)
    skill_df = pd.concat(skill_list)
    assistment_df = pd.concat(assistment_list)
    
    return skill_dict, response_df, skill_df, assistment_df
    
skill_dict, response_df, skill_df, assistment_df = generate_datasets()

with open('skill_dict.json', 'w', encoding='utf-8') as f:
    to_dump_dict = {}
    for key, value in skill_dict.items():
        to_dump_dict[key] = str(value)
    json.dump(to_dump_dict, f)
response_df.to_csv('correct.tsv', sep='\t')
skill_df.to_csv('skill.tsv', sep='\t')
assistment_df.to_csv('assistment_id.tsv', sep='\t')
print('Done')

response_df = pd.read_csv('correct.tsv', sep='\t').drop('Unnamed: 0', axis=1)
skill_df = pd.read_csv('skill.tsv', sep='\t').drop('Unnamed: 0', axis=1)
assistment_df = pd.read_csv('assistment_id.tsv', sep='\t').drop('Unnamed: 0', axis=1)
skill_dict = {}
with open('skill_dict.json', 'r', encoding='utf-8') as f:
    loaded = json.load(f)
    for k, v in loaded.items():
        skill_dict[k] = int(v)

skill_num = len(skill_dict) + 1 # including 0

def one_hot(skill_matrix, vocab_size):
    '''
    params:
        skill_matrix: 2-D matrix (student, skills)
        vocal_size: size of the vocabulary
    returns:
        a ndarray with a shape like (student, sequence_len, vocab_size)
    '''
    seq_len = skill_matrix.shape[1]
    result = np.zeros((skill_matrix.shape[0], seq_len, vocab_size))
    for i in range(skill_matrix.shape[0]):
        result[i, np.arange(seq_len), skill_matrix[i]] = 1.
    return result

def dkt_one_hot(skill_matrix, response_matrix, vocab_size):
    seq_len = skill_matrix.shape[1]
    skill_response_array = np.zeros((skill_matrix.shape[0], seq_len, 2 * vocab_size))
    for i in range(skill_matrix.shape[0]):
        skill_response_array[i, np.arange(seq_len), 2 * skill_matrix[i] + response_matrix[i]] = 1.
    return skill_response_array

def preprocess(skill_df, response_df, skill_num):
    skill_matrix = skill_df.iloc[:, 1:].values
    response_array = response_df.iloc[:, 1:].values
    skill_array = one_hot(skill_matrix, skill_num)
    skill_response_array = dkt_one_hot(skill_matrix, response_array, skill_num)
    return skill_array, response_array, skill_response_array
    

skill_array, response_array, skill_response_array = preprocess(skill_df, response_df, skill_num)

import keras
from keras.layers import Input, Dense, LSTM, TimeDistributed, Lambda, multiply
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

def build_skill2skill_model(input_shape, lstm_dim=32, dropout=0.0):
    input = Input(shape=input_shape, name='input_skills')
    lstm = LSTM(lstm_dim, 
                return_sequences=True, 
                dropout=dropout,
                name='lstm_layer')(input)
    output = TimeDistributed(Dense(input_shape[-1], activation='softmax'), name='probability')(lstm)
    model = Model(inputs=[input], outputs=[output])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def reduce_dim(x):
    x = K.max(x, axis=-1, keepdims=True)
    return x

def build_dkt_model(input_shape, lstm_dim=32, dropout=0.0):
    input_skills = Input(shape=input_shape, name='input_skills')
    lstm = LSTM(lstm_dim, 
                return_sequences=True, 
                dropout=dropout,
                name='lstm_layer')(input_skills)
    dense = TimeDistributed(Dense(int(input_shape[-1]/2), activation='sigmoid'), name='probability_for_each')(lstm)
    
    skill_next = Input(shape=(input_shape[0], int(input_shape[1]/2)), name='next_skill_tested')
    merged = multiply([dense, skill_next], name='multiply')
    reduced = Lambda(reduce_dim, output_shape=(input_shape[0], 1), name='reduce_dim')(merged)
    
    model = Model(inputs=[input_skills, skill_next], outputs=[reduced])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

print('skill2skill')
skill2skill_model = build_skill2skill_model((99, skill_num), lstm_dim=64)

print('dkt')
dkt_model = build_dkt_model((99, 2 * skill_num), lstm_dim=64)
    

# train skill2skill
skill2skill_model.fit(skill_array[:, 0:-1], 
                      skill_array[:, 1:],
                      epochs=20, 
                      batch_size=32, 
                      shuffle=True,
                      validation_split=0.2)

dkt_model.fit([skill_response_array[:, 0:-1], skill_array[:, 1:]],
              response_array[:, 1:, np.newaxis],
              epochs=20, 
              batch_size=32, 
              shuffle=True,
              validation_split=0.2)

sorted_df = df.groupby(by=['skill_name']).count()

most = sorted_df.sort_values(by='order_id', ascending=[False]).index[0:5]
print("5 most common skills are:", [str(skill_dict[skill]) + ": " + skill for skill in most])

least = sorted_df.sort_values(by='order_id', ascending=[True]).index[0:5]
print("5 least common skills are:", [str(skill_dict[skill]) + ": " + skill for skill in least])

most_df = sorted_df.sort_values(by='order_id', ascending=[False])
total = most_df.ix[:,0].sum()
most_common_skill = most_df.iloc[0,0]

most_common_skill / total

X = skill_array[:, 0:-1]
y = skill_array[:, 1:]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, test_size=0.3)

skill2skill_model.fit(X_train, 
                      y_train,
                      epochs=20, 
                      batch_size=32, 
                      shuffle=True,
                      validation_split=0.2)

predictions = skill2skill_model.predict(X_test)
one_hot_predictions = []
for i in np.arange(len(predictions)):
    one_hot_layer = []
    for j in np.arange(len(predictions[0])):
            index_of_max = np.argmax(predictions[i][j])
            one_hot_version = np.zeros(skill_num)
            one_hot_version[index_of_max] = 1
            one_hot_layer.append(one_hot_version)
    one_hot_predictions.append(one_hot_layer)

error_rate = np.count_nonzero(y_test - one_hot_predictions)/2/(y_test.shape[0] * y_test.shape[1])
1-error_rate

import operator

one_hot_predictions - y_test

correct = {}
incorrect = {}

for i in range(len(one_hot_predictions)):
    for j in range(len(one_hot_predictions[0])):
        prediction = one_hot_predictions[i][j]
        actual = y_test[i][j]
        
        comparison = prediction - actual
        position = np.argmax(actual) + 1
        if 1 not in comparison and -1 not in comparison:
            if position in correct.keys():
                correct[position] += 1
            else:
                correct[position] = 1
        else:
            if position in incorrect.keys():
                incorrect[position] += 1
            else:
                incorrect[position] = 1
                

for i in np.arange(1, 112):
    if i not in correct.keys():
        correct[i] = 0
    if i not in incorrect.keys():
        incorrect[i] = 0
        
totals = [correct[i] + incorrect[i] for i in np.arange(1, 112)]
for i in range(1, len(correct)+1):
    if totals[i-1] != 0:
        correct[i] = correct[i] / totals[i-1]
        incorrect[i] = incorrect[i] / totals[i-1]
    else:
        if correct[i] != 0 or incorrect[i] != 0:
            print("something is incorrect lol")

sorted_correct = sorted(correct.items(), key=operator.itemgetter(1))
easiest_to_identify_skills = sorted_correct[-5:]
sorted_incorrect = sorted(incorrect.items(), key=operator.itemgetter(1))
hardest_to_identify_skills = sorted_incorrect[-5:]

easiest_to_identify_skills, hardest_to_identify_skills

def build_betterskill2skill_model(input_shape, lstm_dim=32, dropout=0.0):
    input = Input(shape=input_shape)
    lstm = LSTM(lstm_dim, 
                return_sequences=True, 
                dropout=dropout)(input)
    output = TimeDistributed(Dense(input_shape[-1], activation='softmax'), name='probability')(lstm)
    model = Model(inputs=[input], outputs=[output])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

print('betterskill2skill')
betterskill2skill_model = build_betterskill2skill_model((99, skill_num), lstm_dim=64)

betterskill2skill_model.fit(X_train, 
                            y_train,
                            epochs=200, 
                            batch_size=128, 
                            shuffle=True,
                            validation_split=0.2)

predictions = betterskill2skill_model.predict(X_test)
one_hot_predictions = []
for i in np.arange(len(predictions)):
    one_hot_layer = []
    for j in np.arange(len(predictions[0])):
            index_of_max = np.argmax(predictions[i][j])
            one_hot_version = np.zeros(skill_num)
            one_hot_version[index_of_max] = 1
            one_hot_layer.append(one_hot_version)
    one_hot_predictions.append(one_hot_layer)

error_rate = np.count_nonzero(y_test - one_hot_predictions)/2/(y_test.shape[0] * y_test.shape[1])
error_rate

x = skill_response_array[:, 0:-1]
skill = skill_array[:, 1:]
response = response_array[:, 1:, np.newaxis]

x_train, x_test, skill_train, skill_test, response_train, response_test = model_selection.train_test_split(x, skill, response, train_size=0.7, test_size=0.3)

dkt_model.fit([x_train, skill_train],
              response_train,
              epochs=20, 
              batch_size=32, 
              shuffle=True,
              validation_split=0.2)

dkt_predictions = dkt_model.predict([x_test, skill_test])
for i in np.arange(len(dkt_predictions)):
    for j in np.arange(len(dkt_predictions[0])):
        value = dkt_predictions[i][j][0]
        if value >= 0.5:
            dkt_predictions[i][j][0] = 1
        else:
            dkt_predictions[i][j][0] = 0

error_rate = np.count_nonzero(response_test - dkt_predictions)/2/(response_test.shape[0] * response_test.shape[1])
1-error_rate

from sklearn import metrics 

y_true = response_test.flatten()
y_score = dkt_predictions.flatten()
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
auc = metrics.auc(fpr, tpr)
auc

x = skill_response_array[:, 0:-1]
skill = skill_array[:, 1:]
response = response_array[:, 1:, np.newaxis]

x_train, x_test, skill_train, skill_test, response_train, response_test = model_selection.train_test_split(x, skill, response, train_size=0.7, test_size=0.3)

dkt_model.fit([x_train, skill_train],
              response_train,
              epochs=30, 
              batch_size=38, 
              shuffle=True,
              validation_split=0.2)

dkt_predictions = dkt_model.predict([x_test, skill_test])
for i in np.arange(len(dkt_predictions)):
    for j in np.arange(len(dkt_predictions[0])):
        value = dkt_predictions[i][j][0]
        if value >= 0.5:
            dkt_predictions[i][j][0] = 1
        else:
            dkt_predictions[i][j][0] = 0

error_rate = np.count_nonzero(response_test - dkt_predictions)/2/(response_test.shape[0] * response_test.shape[1])
print("error rate:", error_rate)

y_true = response_test.flatten()
y_score = dkt_predictions.flatten()
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
auc = metrics.auc(fpr, tpr)
print("auc:", auc)

