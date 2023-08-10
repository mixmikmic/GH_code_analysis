import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
#from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix
get_ipython().magic('matplotlib inline')

raw = pd.read_csv("train.csv")
raw.head()

test=raw.sample(frac=0.2,random_state=1251)  #We are creating an 80/20 split, the proportion can be changed by changing the number after  'frac='
train=raw.drop(test.index)
l1 = len(train)
l2 = len(test)
print("Now we have", l1, "training digits and", l2, "testing digits")

train_x = train.iloc[0:,1:]
train_y = train.iloc[0:,:1]
#len(train_x)
#train_x.head()
#train_y.head()
test_x = test.iloc[0:,1:]
test_y = test.iloc[0:,:1]

#The code below allows us to view the image and associated label for each digit
def showNum(row, x, y):
    i=row
    draw = x.iloc[i].as_matrix()
    a = draw.reshape((28,28))
    imgplot = plt.imshow(a)
    plt.title(y.iloc[i])
    
showNum(27, test_x, test_y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
clf.score(test_x, test_y)

train_x[train_x>0]=1
test_x[test_x>0]=1
showNum(27, test_x, test_y)

#Retrain Model...
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
clf.score(test_x, test_y)

pred_y = clf.predict(test_x)
matrix = confusion_matrix(test_y, pred_y)
plt.imshow(matrix, interpolation='none', cmap = 'bwr')
matrix

for i in range(10):
    matrix[i][i] = 0
plt.imshow(matrix, interpolation='none', cmap = 'bwr')
matrix

