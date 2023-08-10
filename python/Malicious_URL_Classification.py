import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
get_ipython().magic('matplotlib inline')

import pickle
X = pickle.load(open("X_data", "rb")) # X_data and y_data should be in the same directory as you notebook.
y = pickle.load(open("y_data", "rb"))
# we initially attempted scaling, but this did not appear to influence the results 
#scaler = StandardScaler(with_mean=False).fit(X)
#X = scaler.transform(X)

#set aside test data
xtrain, xtest_final, ytrain, ytest_final = train_test_split(X, y, test_size=.25)

CVmodel = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', n_jobs=-1).fit(xtrain,ytrain)

Cs = CVmodel.Cs_

scores = CVmodel.scores_[1.0]

cvscores = np.vstack(scores)
lista = np.zeros(10)    #number of values in the CV mse list
for i in cvscores:
    val = 0
    for j in i:
        lista[val] += j
        val += 1
for i in range(len(lista)):
    lista[i] = lista[i] / 5     #divide sums by number of folds
    lista[i] = 1 - lista[i]     #make this mse instead of accuracy 
print(lista)

Cs

import matplotlib.ticker as mtick
Cs = [0.00,0.00,.01,.04, .35,2.78, 21.54, 166.81, 1291.54, 10000.00]   #hard coding this in for the graph
plt.figure(figsize=(7, 5))
plt.xticks(listb, Cs )
plt.ylabel('MSE')
plt.xlabel('C value')
plt.title('Logistic CV Score')
plt.plot(listb, lista)

preds = CVmodel.predict(xtest_final)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

c_mat = confusion_matrix(ytest_final, preds)
plot_confusion_matrix(c_mat)
print("confusion matrix: ")
print(str(c_mat))

(8818+5011)/(48+123+8818+5011)   #percent accuracy

X = pickle.load(open("X_data", "rb"))
y = pickle.load(open("y_data", "rb"))
xtrain, xtest_final, ytrain, ytest_final = train_test_split(X, y, test_size=.25)
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=.25)

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

best_i = None
best_j = None
best_mse = float('inf')
for i in [100, 500, 1000]:
    for j in [100, 500, 1000, None]:
        classifier = RandomForestClassifier(n_estimators = i, n_jobs=3,max_depth = j)
        model = classifier.fit(x_train, y_train)
        predictions = model.predict(x_test)
        predictions = pd.DataFrame(predictions)
        #confusion_matrix = confusion_matrix(y_test, predictions)
        residuals = abs(predictions.as_matrix() - y_test)
        mse = np.mean(residuals)
        if mse < best_mse:
            best_mse - mse
            best_i = i 
            best_j = j
        print (str(mse))


        
print('best n_estimators, depth combo is: (' + str(i) + ', ' + str(j) + ')')

n_features_depth100 = [100, 500, 1000]
n_features_depth500 = [100, 500, 1000]
n_features_depth1000 = [100, 500, 1000]
n_features_depthNone = [100, 500, 1000]
mses100 = [0.065306122449,0.060119047619,0.059693877551]
mses500 = [0.0247448979592,0.0251700680272,0.0268707482993]
mses1000 = [0.0257653061224,0.0269557823129,0.0261904761905]
msesNone = [0.0295918367347,0.0258503401361,0.0255952380952]

plt.figure(figsize=(7, 5))
#plt.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.ylabel('MSE')
plt.xlabel('n_estimators')
plt.title('Random Forest Model Comparisons')
plt.plot(n_features_depth100, mses100, label='depth 100')
plt.plot(n_features_depth500, mses500, label='depth 500')
plt.plot(n_features_depth1000, mses1000, label='depth 1000')
plt.plot(n_features_depthNone, msesNone, label='no max depth')
plt.xticks(n_features_depth100,(100,500,1000))
plt.legend(loc=7)

print(best_i)
print(best_j)

#classifier = RandomForestClassifier(n_estimators = 1000, n_jobs=-1)
classifier = RandomForestClassifier(n_estimators = best_i, max_depth = best_j,n_jobs=-1)
model = classifier.fit(xtrain, ytrain)
ys = pd.DataFrame(ytest_final.tolist())[0]
preds = model.predict(xtest_final)
preds = pd.DataFrame(preds.tolist())[0]
resids = abs(preds - ys)
mse = np.mean(resids)
acc_pct = 1 - mse
print("final mse is: " + str(mse))
print("model is " + str(acc_pct) + "percent accurate")

ys = pd.DataFrame(ytest_final.tolist())[0]
preds = model.predict(xtest_final)
preds = pd.DataFrame(preds.tolist())[0]
resids = abs(preds - ys)
mse = np.mean(resids)
acc_pct = 1 - mse
print("final mse is: " + str(mse))
print("model is " + str(acc_pct) + " percent accurate")

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
preds = model.predict(xtest_final)
c_mat = confusion_matrix(np.asarray(ys), np.asarray(preds))
plot_confusion_matrix(c_mat)
print("confusion matrix: ")
print(str(c_mat))

c_values = [.8, .9, 1, 1.1, 1.2, 1.3]
cv = KFold(n=xtrain.shape[0], n_folds = 5)

mse_store = []  #a list of each fold's mse list.. the value-wise means will be the CV MSEs

for train, test in cv:
    train_x = xtrain[train, :]
    train_y = ytrain[train]
    test_x = xtrain[test, :]
    test_y = ytrain[test]
    mses = []
    for i in range(len(c_values)):
        clf= LinearSVC(C = c_values[i],penalty='l1', loss='squared_hinge', dual=False)
        clf.fit(train_x, train_y) 
        preds = clf.predict(test_x)
        ys = test_y
        ys = pd.DataFrame(ys)
        ys = ys.as_matrix()
        preds = pd.DataFrame(preds)
        preds = preds.as_matrix()
        resids = abs(preds - ys)
        mse = np.mean(resids)
        mses.append(mse)
        print("done with model where C = " + str(c_values[i]) + " with MSE " + str(mse))
    mse_store.append(mses)

#recover and interpret results from cross validation
cvmses = np.vstack(mse_store)
#lista = np.zeros(len(mse_store))
lista = np.zeros(6)    #number of values in the CV mse list
for i in cvmses:
    val = 0
    for j in i:
        lista[val] += j
        val += 1
for i in range(len(lista)):
    lista[i] = lista[i] / 5     #divide sums by number of folds
print(lista)

#get index of min mse. this is the index of the best C value
best_mse = float('inf')
best_c = None   #index of best mse in the list
for i in range(len(lista)):
    if lista[i] < best_mse:
        best_mse = lista[i]
        best_c = i

best_c = c_values[best_c]
best_c

c_values = [.8, .9, 1, 1.1, 1.2, 1.3]
lista = [ 0.01454762,  0.01452381,  0.01466667,  0.01478571,  0.01480952,  0.01478571]
plt.plot(c_values, lista)
plt.ylabel('MSE')
plt.xlabel('C value')
plt.title('SVM CV Score')

clf= LinearSVC(C=best_c,penalty='l1', loss='squared_hinge', dual=False)
clf.fit(xtrain, ytrain)
preds = clf.predict(xtest_final)

ys = pd.DataFrame(ytest_final.tolist())[0]
preds = pd.DataFrame(preds.tolist())[0]
resids = abs(preds - ys)
mse = np.mean(resids)
print("final mse is: " + str(mse))

acc_pct = 1 - mse

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

c_mat = confusion_matrix(ys, preds)
plot_confusion_matrix(c_mat)
print("confusion matrix: ")
print(str(c_mat))

1-mse  #percent accuracy



