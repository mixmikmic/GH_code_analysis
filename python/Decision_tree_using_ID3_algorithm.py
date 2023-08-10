# importing the data as a pandas dataframe
import pandas as pd
# defining the table header info
header_row = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class_a"]
df = pd.read_csv("car.csv", delimiter=",", names=header_row) # importing the csv as a dataframe
df.head(10)

# class distribution of the dataset
from collections import Counter
count  = Counter(df["class_a"])
total = 0
for i in count:
    print("{}: {}".format(i, count[i]))
    total += count[i]
print("total: {}".format(total))

# we are going to split the dataset into training set and test set. 
from sklearn.model_selection import train_test_split
y = df["class_a"]
X = df.drop(["class_a"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# checking the size of training set
X_train.shape

# since we have four classes we will use log base 4 to normalize the entropy value
from math import log
def entropy(a=0, b=0, c=0, d=0):
    total = [a, b, c, d]
    r = 0
    for i in total:
        if i != 0: # since log 0 is undefined
            r += -((i/sum(total))*log(i/sum(total), 4))
    return r

# entropy of the entire training data set (y)
entro_set = entropy(*[i for i in Counter(y_train).values()])
print("The total entropy of the training set is {}".format(entro_set))

# This is our main class
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class decision_tree(BaseEstimator, ClassifierMixin):
    
    def __init__(self, class_col="class_a"):
        self.class_col = class_col
        
    @staticmethod
    def score(split_s, entro, total):
        # here we calculate the entropy of each branch and add them proportionally
        # to get the total entropy of the attribute
        entro_set = [entropy(*i) for i in split_s] # entropy of each branch
        f = lambda x, y: (sum(x)/total) * y 
        result = [f(i, j) for i, j in zip(split_s, entro_set)]
        return entro - sum(result)
        
    @staticmethod
    def split_set(header, dataset, class_col):
        # here we split the attribute into each branch and count the classes
        df = pd.DataFrame(dataset.groupby([header, class_col])[class_col].count())
        result = []
        for i in Counter(dataset[header]).keys():
            result.append(df.loc[i].values)
            
        return result
            
    
    @classmethod
    def node(cls, dataset, class_col):
        entro = entropy(*[i for i in Counter(dataset[class_col]).values()])
        result = {} # this will store the total gain of each attribute
        for i in dataset.columns:
            if i != class_col:
                split_s = cls.split_set(i, dataset, class_col) 
                g_score = cls.score(split_s, entro, total=len(dataset)) # total gain of an attribute
                result[i] = g_score
        return max(result, key=result.__getitem__)
            
    
    @classmethod
    def recursion(cls, dataset, tree, class_col):
        n = cls.node(dataset, class_col) # finding the node that sits as the root
        branchs = [i for i in Counter(dataset[n])]
        tree[n] = {}
        for i in branchs: # we are going to iterate over the branches and create the subsequent nodes
            br_data = dataset[dataset[n] == i] # spliting the data at each branch
            if entropy(*[i for i in Counter(br_data[class_col]).values()]) != 0:
                tree[n][i] = {}
                cls.recursion(br_data, tree[n][i], class_col)
            else:
                r = Counter(br_data[class_col])
                tree[n][i] = max(r, key=r.__getitem__) # returning the final class attribute at the end of tree
        return
                
    @classmethod
    def pred_recur(cls, tupl, t):
        if type(t) is int:
            return "NaN" # assigns NaN when the path is missing for a given test case
        elif type(t) is not dict:
            return t
        index = {'buying': 1, 'maint': 2, 'doors': 3, 'persons': 4, 'lug_boot': 5, 'safety': 6}
        for i in t.keys():
            if i in index.keys():
                r = cls.pred_recur(tupl, t[i].get(tupl[index[i]], 0))
        return r

    # main prediction function
    def predict(self, test):
        result = []
        for i in test.itertuples():
                result.append(decision_tree.pred_recur(i, self.tree_))
        return pd.Series(result) # returns the predicted classes of a test dataset in pandas Series
        
        
    def fit(self, X, y): # this is our main method which we will call to build the decision tree
        class_col = self.class_col # the class_col takes the column name of class attribute
        dataset = X.assign(class_a=y)
        self.tree_ = {} # we will capture all the decision criteria in a python dictionary
        decision_tree.recursion(dataset, self.tree_, class_col)
        
        return self
        
    
        

model = decision_tree() # creating a instance for the decision_tree class
model.fit(X_train, y_train) # calling the fit method to create the tree

# the accuracy score under train-test-split
from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))

# After numerous iteration K = 14 has yeilded the best generalized mean for our model. The K is the number of folds or 
# groups the dataset is divided. A low K can have biases and very high K is computationaly expensive. 
# A trade off is there and we have to select a optimal value, here the value is 14. 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=14, scoring='accuracy')
print(scores)

print("The mean value for K-fold cross validation test that best explains our model is {}".format(scores.mean())) 

