import numpy as np
import pandas as pd
pd.set_option('display.max_rows',1000)

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

random_state = 42

# examine the data file before importing
get_ipython().system('head transfusion.data.txt')

# read data into a Pandas dataframe-- does Pandas figure out the headers?
dat = pd.read_csv('transfusion.data.txt')

print(dat.shape)
dat.head()

# re-do, with shorter names for the columns and skip that first row
cols = ['Recency','Frequency','Monetary','Time','Donated']

dat = pd.read_csv('transfusion.data.txt',names=cols,skiprows=1)

print(dat.shape)
dat.head()

plt.rcParams['figure.figsize'] = (18,10)
dat.plot(subplots=True)
plt.show()

# count the number of 0's and 1's in the response column "Donated"
np.bincount(dat.Donated)

# check the data types
print(dat.dtypes)

# convert the predictors to floats (maybe not necessary, but . . . )
dat[dat.columns[:-1]] *= 1.0
print(dat.dtypes)

dat.isnull().values.any()

predictors = dat[cols[:-1]].copy()   
response   = dat[cols[ -1]].copy()   

predictors.head()

response.head()

predictors.corr()

# visualize the correlation using pyplot.matshow() to display pandas.DataFrame.corr()
def plot_corr(df, size=11):
    """
    Plots correlation matrix for each pair of columns    
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size,size))
    cax = ax.matshow(corr)
    fig.colorbar(cax, fraction=0.0458, pad=0.04)
    plt.xticks(np.arange(len(corr.columns)), corr.columns)
    plt.yticks(np.arange(len(corr.columns)), corr.columns)

plot_corr(predictors, size=5)

# yeah, clearly one column was scaled from the other
set( predictors.Monetary.values / predictors.Frequency.values )

# drop the "Monetary" column
del predictors['Monetary']
predictors.head()

plt.rcParams['figure.figsize'] = (16,8)
predictors.hist()
plt.show()

from sklearn import preprocessing
predictors = preprocessing.scale(predictors)

# package the resulting array back into a DataFrame
predictors = pd.DataFrame(data=predictors,index=np.arange(0,predictors.shape[0]),columns=['Recency','Frequency','Time'])

plt.rcParams['figure.figsize'] = (16,8)
predictors.hist()
plt.show()

# mean of each predictor is 0, std deviation is 1
predictors.describe()

from sklearn.model_selection import train_test_split

test_split_frac = 0.3

X_train, X_test, y_train, y_test = train_test_split(predictors, response, 
                                                            test_size=test_split_frac, random_state=random_state)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# To confirm the split was randomized, note the row indices
X_train.head()

# confirm they match up in the response (no scrambling of indices!)
y_train.head()

# for comparison with the confusion matrix below, count how many actual "0" and "1" responses
# in the test data set
n_true_positive = y_test.sum()
n_true_negative = y_test.shape[0] - y_test.sum()
n_true_positive, n_true_negative

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()  # keep all the defaults 

lr_model.fit(X_train,y_train)

lr_predict_test = lr_model.predict(X_test)
lr_predict_test.shape

# how many incorrectly predicted response values out of 225?
len(np.nonzero( y_test.values - lr_predict_test )[0])

from sklearn import metrics

# "labels" is a list of the response values: 1 or 0; put 1 first in the list to match layout shown above
response_labels = [1,0]

metrics.confusion_matrix(y_test, lr_predict_test, labels=response_labels)

# get recall (and precision) as part of the classification report
print(metrics.classification_report(y_test, lr_predict_test, labels=response_labels))

print("all data fraction of 1's = %.3f" % (      1. * dat.Donated.values.sum() / dat.Donated.values.shape[0] ) )
print("all data fraction of 0's = %.3f" % ( 1. - 1. * dat.Donated.values.sum() / dat.Donated.values.shape[0] ) )

print("y_train fraction of 1's = %.3f" % (      1. * y_train.sum() / y_train.shape[0] ) )
print("y_train fraction of 0's = %.3f" % ( 1. - 1. * y_train.sum() / y_train.shape[0] ) )

# and here's the raw count of 0's and 1's in the training data set
np.bincount(y_train)

lr_model = LogisticRegression(class_weight='balanced')  

lr_model.fit(X_train,y_train)
lr_predict_test = lr_model.predict(X_test)

metrics.confusion_matrix(y_test, lr_predict_test, labels=response_labels)

print(metrics.classification_report(y_test, lr_predict_test, labels=response_labels))

from sklearn.linear_model import LogisticRegressionCV

# cv = 10 means make 10-folds within the Training set
# Cs = 3 means within each fold, make 3 attempts to find best tuning parameter
lr_model_CV = LogisticRegressionCV(Cs=3, cv=10, refit=True, class_weight='balanced')

lr_model_CV.fit(X_train, y_train)

lr_CV_predict = lr_model_CV.predict(X_test)

metrics.confusion_matrix(y_test, lr_CV_predict, labels=response_labels)

print(metrics.classification_report(y_test, lr_CV_predict, labels=response_labels))

