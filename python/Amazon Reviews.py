from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 15)

data = pd.read_csv("Amazon-Reviews.csv")
data = data.loc[:, ['reviews.text', 'reviews.title', 'reviews.rating']]
data

data = data.dropna(thresh=0)
data = data[np.isfinite(data['reviews.rating'])]
data

from matplotlib import pyplot as plt
dist = data.groupby('reviews.rating').count().loc[:, ['reviews.text']].as_matrix()
plt.bar(range(1,6), dist)
plt.xlabel('Star Rating')
plt.ylabel('Frequency')
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

text = np.squeeze(data.loc[:, ['reviews.text']].as_matrix())
title = np.squeeze(data.loc[:, ['reviews.title']].as_matrix())
label = data.loc[:, ['reviews.rating']].as_matrix().reshape(-1)
text_vectoriser = CountVectorizer()
title_vectoriser = CountVectorizer()
text_vector = text_vectoriser.fit_transform(text)
title_vector = title_vectoriser.fit_transform(title)
X = np.concatenate((title_vector.todense(), text_vector.todense()), axis=1)
b = np.ones((1177, 1))
X = np.concatenate((X, b), axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.1)

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

plt.hist(y_train_res)
plt.show()

y_train_res = (np.arange(y_train_res.max()) == y_train_res[:, None]-1).astype(int) #Convert rating into one hot encoding
y_test = (np.arange(y_test.max()) == y_test[:, None]-1).astype(int)

def activate(Q): #A sigmoid
    return 1/(1+np.exp(-Q))

def calc_error(D, Y): #Mean squared error
    return (0.5*np.sum(np.square(D-Y)))/len(D)

#This is our training algorithm, the real bulk of which is just the three
#lines just inside the for loop, the rest is just extra fluff for visualisation
def train_perceptron(X, D, n_epochs, eta, x_test=None, y_test=None):
    W = np.random.uniform(-1, 1, (X.shape[1], D.shape[1]))
    errors = []
    test_errors = []
    for epoch in range(n_epochs):
        Y = activate(np.dot(X, W))
        W += eta*np.dot(((D-Y).T), X).T
        error = calc_error(D, Y)
        errors.append(error)
        #If the test data is supplied, calculate test error too
        if x_test is not None:
            d_test = use_perceptron(x_test, W)
            test_error = calc_error(d_test, y_test)
            test_errors.append(test_error)
    print('Done!')
    
    plt.plot(errors)
    plt.plot(test_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()
    
    return W

#Method that lets us use a trained perceptron for predicting on future data
def use_perceptron(X, W):
    return activate(np.dot(X, W))

n_epochs = 200 #How long to train the network, try adjusting this number! (>200 can take a few seconds)
eta = 0.11 #Our "step size", how big of a step do we take down the hill? Adjust this number too!
W = train_perceptron(x_train_res, y_train_res, n_epochs, eta, x_test=x_test, y_test=y_test)

test_predictions = use_perceptron(x_test, W)

from sklearn.metrics import confusion_matrix, f1_score
y_test_argmax = np.argmax(y_test, axis=1)
test_predictions_argmax = np.argmax(test_predictions, axis=1)
confusion_matrix(y_test_argmax, test_predictions_argmax)

f1 = f1_score(y_test_argmax, test_predictions_argmax, average='weighted')
print(f1)

title_neg = 'Terrible'
text_neg = 'The worst ive ever bought'

title_pos = 'Brilliant'
text_pos = 'The best ive ever bought'

def vectorise(title, text):
    title_vector = title_vectoriser.transform(np.array([title]).ravel())
    text_vector = text_vectoriser.transform(np.array([text]).ravel())
    review_vector = np.concatenate((title_vector.todense(), text_vector.todense()), axis=1)
    b = np.ones((1, 1))
    X = np.concatenate((review_vector, b), axis=1)
    return X

pos_vec = vectorise(title_pos, text_pos)
neg_vec = vectorise(title_neg, text_neg)

W = pickle.load(open('0.71055323893_F1.p', 'rb')) #Load a decent pre-trained model
classification = use_perceptron(pos_vec, W)
print(title_pos, '-', text_pos, ':', str(np.argmax(classification)+1)+" star(s)")
classification = use_perceptron(neg_vec, W)
print(title_neg, '-', text_neg, ':', str(np.argmax(classification)+1)+" star(s)")



