# Import the libraries we will be using
import os
import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy.spatial import distance

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 14, 8

# I've abstracted some of what we'll be doing today into a library.
# You can take a look at this code if you want by going into `dstools/data_tools.py`
from dstools import data_tools

np.random.seed(36)

import urllib.parse
#For python 2 version
#from urlparse import urlparse
from collections import Counter

#pandas version
pd.__version__

# method entropy is used to calculate entropy value for each query
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())


#use pandas data frame for pretty print
normal_usage = pd.read_csv('normal.txt', sep='\t', names=[ "queries"])
df_normal = pd.DataFrame(normal_usage)
df_normal.insert(0, 'label', '0')

#convert original queries to unicode mode
df_normal['unicode'] = df_normal['queries'].str.encode('utf-8')

# add length feature
df_normal['length'] = df_normal['queries'].astype('str').apply(len)
# add entropy feature
df_normal['entropy'] = df_normal['unicode'].map(lambda x: entropy(x))
#summary statistics on normal usage dataset
df_normal.count()

#show variance of normal usage dataset
df_normal.var()

# show df_normal last 5 rows to make sure data is loaded
df_normal.tail()


#use pandas's datagram for better print
malicious_usage = pd.read_csv('malicious.txt', sep='\t', names=[ "queries"])
df_malicious = pd.DataFrame(malicious_usage)
df_malicious.insert(0, 'label', '1')

#convert original queries to 8-bit unicode mode
df_malicious['unicode'] = df_malicious['queries'].str.encode('utf-8')

#add length feature
df_malicious['length'] = df_malicious['queries'].astype('str').apply(len)
#add entropy feature
df_malicious['entropy'] = df_malicious['unicode'].map(lambda x: entropy(x))
#summary statistics on malicious dataset
df_malicious.describe()

#show variance of malicious dataset
df_malicious.var()

#show last 5 rows to make sure data is loaded
df_malicious.tail()

#combine two data frames
data=pd.concat([df_malicious, df_normal],ignore_index=True)
# show statistics 
data.groupby('label').describe()
print (len(data))

#find which query has the largest length and print it
#data['length'].idxmax()
print(data.iloc[215])

#plot length
# set default figure and font size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.size'] = 18.0
plt.rcParams['figure.figsize'] = 15, 5
plt.rcParams['axes.grid'] = True
#plt.ylim[(0,4000)]

#plot historical diagrams by entropy and length
data.hist( column='entropy', by= 'label', bins=150)

data.hist(column='length', by= 'label', bins=150)


# Boxplots show you the distribution of the data (spread).
# http://en.wikipedia.org/wiki/Box_plot

# Plot the length and entropy of web-based statements
data.boxplot('length','label')
plt.ylabel('Statement Length')
data.boxplot('entropy','label')
plt.ylabel('Statement Entropy')


# Split the classes up so we can set colors, size, labels
fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEE5'))
ax.grid(color='grey', linestyle='solid')
cond = data['label'] == 'malicious'
evil = data[cond]
legit = data[~cond]
plt.scatter(df_normal['length'], df_normal['entropy'], s=140, c='#aaaaff', label='Legit', alpha=.7)
plt.scatter(df_malicious['length'], df_malicious['entropy'], s=40, c='r', label='Injections', alpha=.3)
plt.legend()
plt.xlabel('Statement Length')
plt.ylabel('Statement Entropy')

#Knn algorithm
Y = data['label']
data_modify=data.drop(['label','queries','unicode'],axis=1)
X=data_modify
#X = df_normal(['length','entropy'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #splitting data
print (len(y_test))

model = KNeighborsClassifier(1)
model.fit(X_train, y_train)

for k in [1,5,15,50,80,100]:
    model = KNeighborsClassifier(k)
    model.fit(X_train, y_train)
    print ("Accuracy with k = %d is %.3f" % (k, metrics.accuracy_score(y_test, model.predict(X_test)))) 

#SVM
import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl
from sklearn import svm

seaborn.set()

clf1 = svm.SVC(C=1,kernel='linear',gamma=20,decision_function_shape='ovr')
clf2 = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')
#clf3 = svm.SVC(C=1, kernel='poly', degree=3,gamma=20, decision_function_shape='ovr' )
clf3 = svm.SVC(C=1,kernel='sigmoid',gamma=20, decision_function_shape='ovr')
clf4 = svm.LinearSVC(C=1)

clf1=clf1.fit(X_train, y_train)
clf2=clf2.fit(X_train, y_train)
#clf3=clf3.fit(X_train, y_train)

clf3=clf3.fit(X_train, y_train)
#clf4=clf4.fit(X_train, y_train)

# accuracy
y_hat1 = clf1.predict(X_test)
y_hat2 = clf2.predict(X_test)
y_hat3 = clf3.predict(X_test)
#y_hat4 = clf4.predict(X_test)
print ("Linear accuracy with test dataset is %.3f" % metrics.accuracy_score(y_test, y_hat1))
print ("RBF accuracy with test dataset is %.3f" % metrics.accuracy_score(y_test, y_hat2))
print ("Sigmoid accuracy with test dataset is %.3f" % metrics.accuracy_score(y_test, y_hat3))
#print ("LinearSVC accuracy with test dataset is %.3f" % metrics.accuracy_score(y_test, y_hat4))
#show_accuracy(y_hat, y_test, 'test dataset')

h=0.2
x_min, x_max = X_test['length'].min() - 10, X_test['length'].max() + 10  
y_min, y_max = X_test['entropy'].min() - 1, X_test['entropy'].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max,h),  
                     np.arange(y_min, y_max,h)) 
answer1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
answer2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
answer3 = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
#grid_test = np.stack((x1.flat, x2.flat), axis=1)  # test points
#grid_hat = clf1.predict(grid_test)       
#grid_hat = grid_hat.reshape(x1.shape) 
# print 'grid_test = \n', grid_testgrid_hat = clf.predict(grid_test)  

titles = ['LinearSVC (linear kernel)',  
          'SVC with RBF kernel',  
          'SVC with Sigmoid kernel'] 

import matplotlib as mpl
z = answer1.reshape(xx.shape) 
cm_dark1 = mpl.colors.ListedColormap(['w', 'r'])
cm_dark2 = mpl.colors.ListedColormap(['g', 'b'])
plt.contourf(xx, yy, z, cmap=cm_dark2, alpha=0.8)  
plt.scatter(X['length'], X['entropy'], c=Y, cmap=cm_dark1)  
plt.xlabel(u'length')  
plt.ylabel(u'entropy')  
plt.xlim(xx.min(), xx.max())  
plt.ylim(yy.min(), yy.max())  
plt.xticks(())  
plt.yticks(()) 
plt.title(titles[0])  
plt.show()

z = answer2.reshape(xx.shape) 
cm_dark1 = mpl.colors.ListedColormap(['w', 'r'])
cm_dark2 = mpl.colors.ListedColormap(['g', 'b'])
plt.contourf(xx, yy, z, cmap=cm_dark2, alpha=0.8)  
plt.scatter(X['length'], X['entropy'], c=Y, cmap=cm_dark1)  
plt.xlabel(u'length')  
plt.ylabel(u'entropy')  
plt.xlim(xx.min(), xx.max())  
plt.ylim(yy.min(), yy.max())  
plt.xticks(())  
plt.yticks(()) 
plt.title(titles[1])  
plt.show()

z = answer3.reshape(xx.shape) 
cm_dark1 = mpl.colors.ListedColormap(['w', 'r'])
cm_dark2 = mpl.colors.ListedColormap(['g', 'b'])
plt.contourf(xx, yy, z, cmap=cm_dark2, alpha=0.8)  
plt.scatter(X['length'], X['entropy'], c=Y, cmap=cm_dark1)  
plt.xlabel(u'length')  
plt.ylabel(u'entropy')  
plt.xlim(xx.min(), xx.max())  
plt.ylim(yy.min(), yy.max())  
plt.xticks(())  
plt.yticks(()) 
plt.title(titles[2])  
plt.show()

# may add more feature
#print (data['unicode'])

def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r',encoding="utf-8") as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result
############################################################
#### Use python script directly for efficency 
############################################################


badQueries = loadFile('malicious.txt')
validQueries = loadFile('normal.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))


allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]

y = yBad + yGood #y equals to total labels included bad and good queries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split


#transform data into a TFIDF matrix of 1-gram and 2-gram

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,5)) #converting data to vectors
x = vectorizer.fit_transform(allQueries)

X_transform=x.toarray()
feature_name = vectorizer.get_feature_names()
newDataFrame = pd.DataFrame(X_transform, columns = feature_name)


badCount = len(badQueries)
validCount = len(validQueries)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10) #splitting data
lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}) # class_weight='balanced')




lgs.fit(X_train, y_train) #training our model
print (x.shape)
print(lgs.score(X_test, y_test))

predicted = lgs.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + badCount)))
print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)

from sklearn.metrics import classification_report




