# import all dependencies
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from attention_decoder import AttentionDecoder
from nltk.stem import PorterStemmer
from tqdm import tqdm
from helper import *
import string
import pandas as pd

# load csv into pandas dataframe
df = pd.read_csv('veoh_qna.csv')
df.head(10)

# create an index of words, please note that index 0 is set to an empty space
t_df = df_to_df(df.question, df.answer)
t_df.head()

# transform questions & answers to a word array and a sequence array
q_list, q_as_array = word_to_array(df.question, t_df)
a_list, a_as_array = word_to_array(df.answer, t_df)

# print the first array
print('Question word list:\n', q_list[:1], '\n'*2,'Question array list:\n', q_as_array[:1], '\n'*2)
print('Answer word list:\n', a_list[:1],'\n'*2, 'Answer array list:\n', a_as_array[:1],'\n')

# use the length of the index of the word matrix as the vocabulary size
vocab_size = len(t_df)
print('Vocab Size: ', vocab_size)

# set max features(vocab size) equal to vocab size
n_features = vocab_size
print('Number of features: ', n_features, '\n')

# find the max length of question & answer
max_q_l = len(max(q_as_array,key=len))
max_a_l = len(max(a_as_array,key=len))
max_l = max(max_q_l, max_a_l)
print('Max Length of Question: ', max_q_l)
print('Max Length of Answer: ', max_a_l)

# set max length equal to max length + 3 to ensure ample padding
max_length = max_l + 3
print('Max Padded Length: ', max_length)

# using the keras function pad_sequences, we pad with the default value of 0 up to the max length of any q&a

# pad questions to max length
padded_q_docs = pad_sequences(q_as_array, maxlen=max_length, padding='post')
print('Padded questions array:\n', padded_q_docs[:1])

# pad answers to max length
padded_a_docs = pad_sequences(a_as_array, maxlen=max_length, padding='post')
print('\nPadded answers array:\n', padded_a_docs[:1])

# define model
model = Sequential()
model.add(LSTM(150, input_shape=(max_length, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_features))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train the model for 40% of the length for number of features
for a in tqdm(range(0, n_features//10*4)):
    for n in range(0, len(padded_q_docs)):
        # transform xy
        X,y = transform_xy(padded_q_docs[n], padded_a_docs[n], n_features)
        
        # fit model for one epoch on this sequence
        model.fit(X, y, epochs=1, verbose=0)

# print 3 sets of questions, expected response and predicted response
for n in range(10, 12):
    X,y = transform_xy(padded_q_docs[n], padded_a_docs[n], n_features)
    yhat = model.predict(X, verbose=0)
    print('Set #{}'.format(n))
    print('Question Array:', one_hot_decode(X[0]), '\nQuestion :', array_to_string(one_hot_decode(X[0]), t_df), '\n')
    print('Expected Response Array:', one_hot_decode(y[0]), '\nExpected Response:', array_to_string(one_hot_decode(y[0]), t_df), '\n')
    print('Predicted Response Array:', one_hot_decode(yhat[0]), '\nPredicted Response:', array_to_string(one_hot_decode(yhat[0]), t_df), '\n')

# print accuracy of model
total, correct = len(padded_q_docs), 0
for n in range(total):
    X,y = transform_xy(padded_q_docs[n], padded_a_docs[n], n_features)
    yhat = model.predict(X, verbose=0)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Total Training Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# create sentences that are not in the list of questions and answers list
sent0 = "hi"
sent1 = "how are you?"
sent2 = "how can i download some videos?"
sent3 = "where do i upload a video?"
sent4 = "what file formats do you recommend for uploads?"
sent5 = "is there a size limit for uploading?"
sent6 = "how can i get better search results?"
sent7 = "what is veoh compass?"
sent8 = "where can i search for videos or groups?"
sent9 = "thank you"
sent10 = "bye"

for n in range(0, 11):
    print('\nUser: ', eval('sent'+str(n)),
          '\nVeoh Bot: ', get_response(eval('sent'+str(n)), t_df, max_length, n_features, model))

