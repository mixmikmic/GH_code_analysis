get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

df_train = pd.read_csv("/resources/data/brainwaves/predict_fraudulant/train.csv")
df_test =  pd.read_csv("/resources/data/brainwaves/predict_fraudulant/test.csv")

df_train

#joining test and train datasets
df= pd.concat([df_train, df_test], axis=0, join='outer')

#cat_var_1 and cat_var_6 have small no of nans.....imputing them
df['cat_var_1'].fillna(df['cat_var_1'].value_counts().idxmax(), inplace=True)
df['cat_var_6'].fillna(df['cat_var_6'].value_counts().idxmax(), inplace=True)
df.isnull().sum()

df['cat_var_3'].fillna(df['cat_var_3'].value_counts().idxmax(), inplace=True)

#dropping all redundant features
df=df.drop(['cat_var_31', 'cat_var_33', 'cat_var_34', 'cat_var_35', 'cat_var_36'
            ,'cat_var_37', 'cat_var_38', 'cat_var_40', 'cat_var_41', 'cat_var_42'
            ,'cat_var_8'], axis=1,)
df.isnull().sum()

label_enc = LabelEncoder()
df['cat_var_1']=label_enc.fit_transform(df['cat_var_1'])
df['cat_var_2']=label_enc.fit_transform(df['cat_var_2'])
df['cat_var_3']=label_enc.fit_transform(df['cat_var_3'])
df['cat_var_4']=label_enc.fit_transform(df['cat_var_4'])
df['cat_var_5']=label_enc.fit_transform(df['cat_var_5'])
df['cat_var_6']=label_enc.fit_transform(df['cat_var_6'])
df['cat_var_7']=label_enc.fit_transform(df['cat_var_7'])
df['cat_var_9']=label_enc.fit_transform(df['cat_var_9'])
df['cat_var_10']=label_enc.fit_transform(df['cat_var_10'])
df['cat_var_11']=label_enc.fit_transform(df['cat_var_11'])
df['cat_var_12']=label_enc.fit_transform(df['cat_var_12'])
df['cat_var_13']=label_enc.fit_transform(df['cat_var_13'])
df['cat_var_14']=label_enc.fit_transform(df['cat_var_14'])
df['cat_var_15']=label_enc.fit_transform(df['cat_var_15'])
df['cat_var_16']=label_enc.fit_transform(df['cat_var_16'])
df['cat_var_17']=label_enc.fit_transform(df['cat_var_17'])
df['cat_var_18']=label_enc.fit_transform(df['cat_var_18'])
df

df_train =df.iloc[:348978,:]
df_test =df.iloc[348978:,:]
df_test.head()

X = df_train.loc[:,['cat_var_1',
 'cat_var_10', 'cat_var_11', 'cat_var_12', 'cat_var_13', 'cat_var_14',
 'cat_var_15', 'cat_var_16', 'cat_var_17', 'cat_var_18', 'cat_var_19',
 'cat_var_2', 'cat_var_20', 'cat_var_21', 'cat_var_22', 'cat_var_23',
 'cat_var_24', 'cat_var_25', 'cat_var_26', 'cat_var_27', 'cat_var_28',
 'cat_var_29', 'cat_var_3', 'cat_var_30', 'cat_var_32', 'cat_var_39', 'cat_var_4',
 'cat_var_5', 'cat_var_6', 'cat_var_7', 'cat_var_9', 'num_var_1', 'num_var_2',
 'num_var_3', 'num_var_4', 'num_var_5', 'num_var_6', 'num_var_7']]

Y = df_train.loc[:, 'target']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y,
                                        test_size=0.33, random_state=7)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=38, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(15,kernel_initializer='normal',activation = 'sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(10,kernel_initializer='normal',activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=20, batch_size=10, verbose=1, )))
pipeline = Pipeline(estimators)

pipeline.fit(X_train,Y_train)

s_pred = pipeline.predict_proba(df_test.loc[:,['cat_var_1',
 'cat_var_10', 'cat_var_11', 'cat_var_12', 'cat_var_13', 'cat_var_14',
 'cat_var_15', 'cat_var_16', 'cat_var_17', 'cat_var_18', 'cat_var_19',
 'cat_var_2', 'cat_var_20', 'cat_var_21', 'cat_var_22', 'cat_var_23',
 'cat_var_24', 'cat_var_25', 'cat_var_26', 'cat_var_27', 'cat_var_28',
 'cat_var_29', 'cat_var_3', 'cat_var_30', 'cat_var_32', 'cat_var_39', 'cat_var_4',
 'cat_var_5', 'cat_var_6', 'cat_var_7', 'cat_var_9', 'num_var_1', 'num_var_2',
 'num_var_3', 'num_var_4', 'num_var_5', 'num_var_6', 'num_var_7']])

s_pred

c=0
s_test=[]
for arr in s_pred:
    s_test.append(arr[1])
    c=c+1
s_test

df_test['target'] = s_test
df_pred = df_test.loc[:,['target', 'transaction_id']]
df_pred.to_csv('/resources/data/brainwaves/predict_fraudulant/pred_fraud_nn1.csv')



