#Import libraries

import numpy as np
import pandas as pd
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import cluster
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing

# Compute confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

dbbig = pd.read_csv("../data/movie_ratings_simple.csv")

genres=pd.concat([dbbig['Genre1'],dbbig['Genre2'],dbbig['Genre3']])

#Try model fitting on just director, year, and genre and see what we get.

#Get a reduced dataframe with the stuff we need
#dfred = dbbig[["Director1","year","Genre1",'Genre2','Genre3']]
dfred = pd.DataFrame()
#This needs to be re-shaped. There should be a single "Genre" category and movies that have multiple Genres should have multiple
y=(dbbig["stars"].reset_index())['stars']

leD = preprocessing.LabelEncoder()
leD.fit(dbbig["Director1"])
dfred['Director']=leD.transform(dbbig["Director1"])

leG = preprocessing.LabelEncoder()
leG.fit(genres)
dfred['Genre1'] = leG.transform(dbbig['Genre1'])
dfred['Genre2'] = leG.transform(dbbig['Genre2'])
dfred['Genre3'] = leG.transform(dbbig['Genre3'])

leY = preprocessing.LabelEncoder()
leY.fit(dbbig['year'])
dfred['year'] = leY.transform(dbbig['year'])


#How well does a decision tree model work to predict these data?
X_train, X_test, y_train, y_test = train_test_split(dfred,y,test_size=0.10, random_state=1) #split the data for training

#dtc = tree.DecisionTreeClassifier(min_samples_leaf=2)
dtc = ensemble.RandomForestClassifier(n_estimators=200,min_samples_leaf=5)
#dtc = svm.SVC(kernel='rbf',gamma=10,C=100,probability=True)
dtc.fit(X_train,y_train)
print( "score = {}".format(dtc.score(X_test,y_test)))
print (dtc.feature_importances_)
pdata =dtc.predict(X_test)

#dtc_probs = dtc.predict_proba(X_test)
#score = log_loss(y_test, dtc_probs)
#print( "logloss = {}".format(score))
cm = confusion_matrix(y_test, pdata)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

