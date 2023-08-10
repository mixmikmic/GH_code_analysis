#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('Customers_Bank_Accounts.csv')

#inspect dataset
df.shape
df.head(5)

df.describe()
df.isnull().sum()

import seaborn as sns
corr = df.corr()
sns.heatmap(corr, cmap = 'YlGnBu')
plt.show()

#drop columns = customerID, Row Number and Surname
X = df.iloc[:, 3:13].values  
Y = df.iloc[:, 13].values  

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # encoding countries 
labelencoder_X_2 = LabelEncoder() 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # encoding gender
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# avoid dummy variable trap - remove first column generated from countries encoding
X = X[:, 1:] 

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler ()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

import keras
from keras.models import Sequential 
from keras.layers import Dense

classifier = Sequential()  

classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))

# to improve model's accuracy 
classifier.add(Dense(units = 6, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Callback
from keras.callbacks import History
histories = History()

# FItting the ANN to the Training set 
### Run a batch size of 32 observations before all the weights are updated. 
classifier.fit(X_train,Y_train, batch_size = 32, epochs = 150, verbose = 0, validation_data = (X_test,Y_test),callbacks = [histories])

score = classifier.evaluate(X_test, Y_test, verbose=0)
print('\nThe {0} function of the test set is: {1:0.3}'.format(classifier.metrics_names[0],score[0]))
print('The {0} of the test set is: {1:0.3%}'.format(classifier.metrics_names[1],score[1]))

score = classifier.evaluate(X_train, Y_train, verbose=0)
print('\nThe {0} function of the training set is: {1:0.3}'.format(classifier.metrics_names[0],score[0]))
print('The {0} of the training set is: {1:0.3%}'.format(classifier.metrics_names[1],score[1]))

##list all the keys in history 
print(histories.history.keys())

# summarize history for accuracy
plt.plot(histories.history['acc'])
plt.plot(histories.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(histories.history['loss'])
plt.plot(histories.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test) # predict y based on X_test (2000 test samples)
y_pred = (y_pred > 0.5) # set a theshold, TRUE = customer will leave the bank

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred) 
cm

y_pred_new_client = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))) 
print ('\nThe probability of this new customer will leave the bank is {:0.2f}%'.format (y_pred_new_client[0,0]*100)) 

from keras.models import Sequential 
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()  
    classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))  
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 100, verbose = 0)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()
print ('\n Mean accuracy: {0:0.2f}% with a variance of {1:0.2f}%'.format(accuracies.mean()*100,accuracies.std()*100))

# Parameter Tuning the ANN: Apply Grid Search 
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV

def build_classifier(my_optimizer):
    classifier = Sequential()  
    classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))  
    classifier.compile(optimizer = my_optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# set up parameters in dictionary for gridsearch 
# Based on the recall results above, we would try the following hyperparameters
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'my_optimizer' : ['adam', 'rmsprop']
              }
# set up gridsearch 
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10
                           )
# fit gridsearch to the training set
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

