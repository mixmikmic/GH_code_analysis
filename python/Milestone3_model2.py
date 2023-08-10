import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss

# import the dataset, using Pandas.
keywords_data = pd.read_csv("./new3.csv")

#### Keep only the keyword columns #########

# drop imdb features 
imdb_cols = list(range(12030,12045))
data = keywords_data.drop(keywords_data.columns[imdb_cols],axis=1)

# drop tmdb features 
tmdb_cols =  list(range(2,16))
data = data.drop(data.columns[tmdb_cols],axis=1)
data = data.drop(data.columns[0],axis=1)

data.head()

#### create training and testing sets
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

# select keyword columns
col_names= list(train1.columns.values)
#this is columns 16 to 12045
keyword_columns= col_names[2:-7]

#want cols Res to be genre type
colsRes1 = ['group1']
colsRes2 = ['group2']
colsRes3 = ['group3']
colsRes4 = ['group4']
colsRes5 = ['group5']
colsRes6 = ['group6']
colsRes7 = ['group7']

trainArr = train1.as_matrix(keyword_columns) #training array
trainRes1 = train1.as_matrix(colsRes1) 
trainRes2 = train1.as_matrix(colsRes2)
trainRes3 = train1.as_matrix(colsRes3)
trainRes4 = train1.as_matrix(colsRes4)
trainRes5 = train1.as_matrix(colsRes5)
trainRes6 = train1.as_matrix(colsRes6)
trainRes7 = train1.as_matrix(colsRes7)

trainsets = [trainRes1, trainRes2, trainRes3, trainRes4, trainRes5, trainRes6, trainRes7]

# select keyword columns
col_names= list(train.columns.values)
#this is columns 16 to 12045
keyword_columns= col_names[2:-7]

X_train = train.as_matrix(keyword_columns) #training array
X_test = test.as_matrix(keyword_columns) #training array

for i in range(1,8):
    
    print('\nGENRE ', i, "\n=====================\n")
    Y_train = train['group' + str(i)].values
    
    rf = RandomForestClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=20, max_depth=5)
    clf.fit(X_train, Y_train)
    
    # extract out feature importances
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    # plot the feature importances of the forest
    plt.title("Feature importances for genre " + str(i))
    plt.bar(range(20), importances[indices[:20]],
       color="r", yerr=std[indices[:20]], align="center")
    plt.xticks(range(20), indices, )
    plt.show()
    
    # print the feature ranking
    print("Feature ranking:")
    for indx, i in enumerate(indices[:10]):
        print (indx+1, ": ", keyword_columns[i])

X_test = test.as_matrix(keyword_columns)
Y_test = test.as_matrix(col_names[-7:])
Y_train = train.as_matrix(col_names[-7:])

## Prediction 
rf = RandomForestClassifier(n_estimators=100)
y_test_pred = OneVsRestClassifier(rf).fit(X_train, Y_train).predict(X_test)

# These are how we measure error - Haming Loss, % exact matches and % at-least-one match
def error_measures(ypred, ytest):
    ypred = np.array(ypred)
    ytest = np.array(ytest)
    # Hamming loss
    from sklearn.metrics import hamming_loss
    h_loss = hamming_loss(ytest, ypred)

    # Percent exact matches
    y_pred_str = np.array([str(yi) for yi in ypred])
    y_test_str = np.array([str(yi) for yi in ytest])
    percent_exact = np.sum(y_pred_str == y_test_str) * 1. / ytest.shape[0]
    
    # Percent at least one match (at least one of the genres are both 1)
    atleastone_count = 0
    for ind in range(len(ypred)):
        yi_pred = ypred[ind]
        yi_test = ytest[ind]
        for i in range(len(yi_pred)):
            if yi_pred[i] == 1 and yi_test[i] == 1:
                atleastone_count += 1
                break
    percent_atleastone = atleastone_count * 1. / ytest.shape[0]
    
    return h_loss, percent_exact, percent_atleastone

error_measures(Y_test, y_test_pred)



