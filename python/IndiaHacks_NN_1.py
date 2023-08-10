## importing the pre-requisites for data handling 

import numpy as np
import pandas as pd

## importing pre-requisites for building Neural Net 

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.preprocessing import StandardScaler,LabelEncoder
from keras.utils.np_utils import to_categorical as tocat

# to initialize psuedorandom number generator (can work as key to generate same results)

np.random.seed(10)

# lets load the datsets

train=pd.read_csv('train.csv') #training 
test=pd.read_csv('test.csv') # real Testing!

# encoding string values to 0,1,2,3  and storing Actuals in y_train

le=LabelEncoder().fit(train.DetectedCamera)
y_train=le.transform(train['SignFacing (Target)'])

# renaming llooooonnngg column names

train.rename(columns={'SignFacing (Target)': 'Actual', 'DetectedCamera': 'Detected'}, inplace=True)

# removing useless columns(Id,Actual) and encoding Detected camera value to 0,1,2,3

X_train = train.drop(['Id','Actual'],axis=1)
X_train['Detected'] = le.transform(X_train['Detected'])
X_train.head()

# scaling the numerical columns thus obtained to improve prediction results
## converting y_train to One-Hot Encoding for probablistic values 

scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
y_train = tocat(y_train)

# Let's have a look at the test data for the first time 

test.head()

# Removing useless columns(Id) and transforming Detected values to corresponding numerical counterparts 0,1,2,3

Id = test.Id
test['DetectedCamera'] = le.transform(test['DetectedCamera'])
X_test = test.drop(['Id'],axis=1)
X_test.shape

# Lets scale the test data now to proportion for achieving better results

X_test=scaler.transform(X_test)

#model begins


model = Sequential()

#layer 1

model.add(Dense(512,input_dim=5))
model.add(Activation("relu"))
model.add(Dropout(0.1))

#layer 2

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))

#layer 2

model.add(Dense(64))
model.add(Activation("sigmoid"))
model.add(Dropout(0.2))

#layer 4

model.add(Dense(4))
model.add(Activation("softmax"))

## compiling the model

### optimizer used ins rmsprop (adagrad is also an optimizer)
### since we are working in multinomial classification problem loss must be calculated using categorical_crossentropy

model.compile(optimizer = "rmsprop",loss='categorical_crossentropy',metrics=['accuracy'])

#lets run the model built

## iterations = 100
## each iteration batch_size is 100
## set verbose=1 for checking the training status


params = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=0)

# Predict the results for this learned model

y_test = model.predict_proba(X_test)

# submit the results in submissionNN.csv
## Job Done!

submission=pd.DataFrame(y_test,index=Id,columns=le.classes_)
submission.to_csv('submission_NN.csv')

