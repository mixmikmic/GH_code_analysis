import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

p = np.linspace(0.01,0.99,100)
plt.figure(figsize=(16,8))
plt.plot(p,1-p**2-(1-p)**2)
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(16,8))
plt.plot(p,-p*np.log2(p)-(1-p)*np.log2(1-p))
plt.ylim(0,1)
plt.show()

from sklearn.datasets import load_iris,load_boston
iris = load_iris()
boston = load_boston()

import pandas as pd
bostonX = boston['data']
bostonY = boston['target']
bostonDF = pd.DataFrame(data = np.hstack((bostonX,bostonY.reshape(bostonY.shape[0],1))),                        columns=np.append(boston['feature_names'],'PRICE'))
bostonDF.head()

irisX = iris['data']
irisY = iris['target']
irisDF = pd.DataFrame(data = np.hstack((irisX,irisY.reshape(irisY.shape[0],1))),                        columns=np.append(iris['feature_names'],"Species"))
irisDF.head()

def entropy(y):
    if y.size == 0: return 0
    p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
    return -1 * np.sum(p * np.log2(p+1e-9))

def gini_impurity(y):
    if y.size == 0: return 0
    p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
    return 1 - np.sum(p**2)

def variance(y):
    if y.size == 0: return 0
    return np.var(y)

print entropy(irisY)
print gini_impurity(irisY)
print variance(bostonY)

def information_gain(y,mask,func=entropy):
    s1 = np.sum(mask)
    s2 = mask.size - s1
    if (s1 == 0 | s2 == 0): return 0
    return func(y) - s1/float(s1+s2) * func(y[mask]) - s2/float(s1+s2) * func(y[np.logical_not(mask)])


print information_gain(irisY,irisX[:,2] < 3.5)
print information_gain(irisY,irisX[:,2] < 3.5,gini_impurity)
print information_gain(bostonY,bostonX[:,5] < 7)

np.apply_along_axis(lambda x: np.sum(x),0,irisX)

def max_information_gain_split(y,x,func=gini_impurity):
    best_change = None
    split_value = None
    is_numeric = irisX[:,2].dtype.kind not in ['S','b']
    
    for val in np.unique(np.sort(x)):
        mask = x == val
        if(is_numeric): mask = x < val
        change = information_gain(y,mask,func)
        if best_change is None:
            best_change = change
            split_value = val
        elif change > best_change:
            best_change = change
            split_value = val
            
    return {"best_change":best_change,            "split_value":split_value,            "is_numeric":is_numeric}

print(max_information_gain_split(irisY,irisX[:,2]))
print(max_information_gain_split(irisY,irisX[:,2],entropy))
print(max_information_gain_split(bostonY,bostonX[:,3],variance))

def best_feature_split(X,y,func=gini_impurity):
    best_result = None
    best_index = None
    for index in range(X.shape[1]):
        result = max_information_gain_split(y,X[:,index],func)
        if best_result is None:
            best_result = result
            best_index = index
        elif best_result['best_change'] < result['best_change']:
            best_result = result
            best_index = index
    
    best_result['index'] = best_index
    return best_result

print best_feature_split(irisX,irisY)
print best_feature_split(irisX,irisY,entropy)
print best_feature_split(bostonX,bostonY,variance)

def get_best_mask(X,best_feature_dict):
    best_mask = None
    if best_feature_dict['is_numeric']:
        best_mask = X[:,best_feature_dict['index']] < best_feature_dict['split_value']
    else:
        best_mask = X[:,best_feature_dict['index']] == best_feature_dict['split_value']
    return best_mask

bfs = best_feature_split(irisX,irisY)
best_mask = get_best_mask(irisX,bfs)
left = irisX[best_mask,:]
right = irisX[np.logical_not(best_mask),:]

class DecisionTreeNode(object):
    
    def __init__(self,            X,            y,            minimize_func,            min_information_gain=0.01,             max_depth=3,             depth=0):
        self.X = None
        self.y = None
        
        self.minimize_func=minimize_func
        self.min_information_gain=min_information_gain
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.depth = depth
        
        self.best_split = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.split_description = ""
        
    def _information_gain(self,mask):
        pass
    
    def _max_information_gain_split(self,X):
        pass
    
    def _best_feature_split(self):
        pass
    
    def _split_node(self):
        pass
    
    def _predict_row(self,row):
        pass
    
    def predict(self,X):
        pass
    
    def __repr__(self):
        pass

class DecisionTreeNode(object):
    
    def __init__(self,            X,            y,            minimize_func,            min_information_gain=0.001,            max_depth=3,            min_leaf_size=20,            depth=0):
        self.X = X
        self.y = y
        
        self.minimize_func=minimize_func
        self.min_information_gain=min_information_gain
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.depth = depth
        
        self.best_split = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.split_description = "root"
        
    def _information_gain(self,mask):
        s1 = np.sum(mask)
        s2 = mask.size - s1
        if (s1 == 0 | s2 == 0): return 0
        return self.minimize_func(self.y) -                 s1/float(s1+s2) * self.minimize_func(self.y[mask]) -                 s2/float(s1+s2) * self.minimize_func(self.y[np.logical_not(mask)])
    
    def _max_information_gain_split(self,x):
        best_change = None
        split_value = None
        previous_val = None
        is_numeric = x.dtype.kind not in ['S','b']

        for val in np.unique(np.sort(x)):
            mask = x == val
            if(is_numeric): mask = x < val
            change = self._information_gain(mask)
            s1 = np.sum(mask)
            s2 = mask.size-s1
            
            if best_change is None and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
                best_change = change
                split_value = val
            elif change > best_change and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
                best_change = change
                split_value = np.mean([val,previous_val])
            
            previous_val = val

        return {"best_change":best_change,                "split_value":split_value,                "is_numeric":is_numeric}
    
    def _best_feature_split(self):
        best_result = None
        best_index = None
        for index in range(self.X.shape[1]):
            result = self._max_information_gain_split(self.X[:,index])
            if result['best_change'] is not None:
                if best_result is None:
                    best_result = result
                    best_index = index
                elif best_result['best_change'] < result['best_change']:
                    best_result = result
                    best_index = index
        
        if best_result is not None:
            best_result['index'] = best_index
            self.best_split = best_result
    

    
    def _split_node(self):
        
        if self.depth < self.max_depth :
            
            self._best_feature_split() 
            
            if self.best_split is not None and self.best_split['best_change'] >= self.min_information_gain :
                   
                mask = None
                if self.best_split['is_numeric']:
                    mask = self.X[:,self.best_split['index']] < self.best_split['split_value']
                else:
                    mask = self.X[:,self.best_split['index']] == self.best_split['split_value']
                
                if(np.sum(mask) >= self.min_leaf_size and (mask.size-np.sum(mask)) >= self.min_leaf_size):
                    self.is_leaf = False
                    
                    self.left = DecisionTreeNode(self.X[mask,:],                                                self.y[mask],                                                self.minimize_func,                                                self.min_information_gain,                                                self.max_depth,                                                self.min_leaf_size,                                                self.depth+1)

                    if self.best_split['is_numeric']:
                        split_description = 'index ' + str(self.best_split['index']) + " < " + str(self.best_split['split_value']) + " ( " + str(self.X[mask,:].shape[0]) + " )"
                        self.left.split_description = str(split_description)
                    else:
                        split_description = 'index ' + str(self.best_split['index']) + " == " + str(self.best_split['split_value']) + " ( " + str(self.X[mask,:].shape[0]) + " )"
                        self.left.split_description = str(split_description)

                    self.left._split_node()
                    
                    
                    self.right = DecisionTreeNode(self.X[np.logical_not(mask),:],                                                self.y[np.logical_not(mask)],                                                self.minimize_func,                                                self.min_information_gain,                                                self.max_depth,                                                self.min_leaf_size,                                                self.depth+1)
                    
                    if self.best_split['is_numeric']:
                        split_description = 'index ' + str(self.best_split['index']) + " >= " + str(self.best_split['split_value']) + " ( " + str(self.X[np.logical_not(mask),:].shape[0]) + " )"
                        self.right.split_description = str(split_description)
                    else:
                        split_description = 'index ' + str(self.best_split['index']) + " != " + str(self.best_split['split_value']) + " ( " + str(self.X[np.logical_not(mask),:].shape[0]) + " )"
                        self.right.split_description = str(split_description)

                   
                    self.right._split_node()
                    
        if self.is_leaf:
            if self.minimize_func == variance:
                self.split_description = self.split_description + " : predict - " + str(np.mean(self.y))
            else:
                values, counts = np.unique(self.y,return_counts=True)
                predict = values[np.argmax(counts)]
                self.split_description = self.split_description + " : predict - " + str(predict)
                                          
    
    def _predict_row(self,row):
        predict_value = None
        if self.is_leaf:
            if self.minimize_func==variance:
                predict_value = np.mean(self.y)
            else:
                values, counts = np.unique(self.y,return_counts=True)
                predict_value = values[np.argmax(counts)]
        else:
            left = None
            if self.best_split['is_numeric']:
                left = row[self.best_split['index']] < self.best_split['split_value']
            else:
                left = row[self.best_split['index']] == self.best_split['split_value']
                
            if left:
                predict_value = self.left._predict_row(row)
            else:
                predict_value = self.right._predict_row(row)
 
        return predict_value
    
    def predict(self,X):
        return np.apply_along_axis(lambda x: self._predict_row(x),1,X)
    
    def _rep(self,level):
        response = "|->" + self.split_description
        
        if self.left is not None:
            response += "\n"+(2*level+1)*" "+ self.left._rep(level+1)
        if self.right is not None:
            response += "\n"+(2*level+1)*" "+ self.right._rep(level+1)
        
        return response
    
    def __repr__(self):
        return self._rep(0)
        

dtn = DecisionTreeNode(irisX,irisY,gini_impurity)
dtn._split_node()
dtn

dtn = DecisionTreeNode(bostonX,bostonY,variance,max_depth=5)
dtn._split_node()
dtn

class DecisionTree(object):
    
    def __init__(self,            minimize_func,            min_information_gain=0.01,            max_depth=3,            min_leaf_size=20):
        
        self.root = None
        self.minimize_func = minimize_func
        self.min_information_gain = min_information_gain
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        
    def fit(X,y):
        pass
    
    def predict(X):
        pass

class DecisionTree(object):
    
    def __init__(self,            minimize_func,            min_information_gain=0.001,            max_depth=3,            min_leaf_size=20):
        
        self.root = None
        self.minimize_func = minimize_func
        self.min_information_gain = min_information_gain
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        
    def fit(self,X,y):
        self.root =  DecisionTreeNode(X,                                    y,                                    self.minimize_func,                                    self.min_information_gain,                                    self.max_depth,                                    self.min_leaf_size,                                    0)
        self.root._split_node()
    
    def predict(self,X):
        return self.root.predict(X)
    
    def __repr__(self):
        return self.root._rep(0)

from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.tree import DecisionTreeRegressor as SklearnDTR
from sklearn.tree import _tree

def print_sklearn_tree(tree,index=0,level=0):
    response = ""
    if level == 0:
        response += "root\n"
    

    if tree.feature[index] == -2:
        response +=  ": predict " + str(np.argmax(dt_sklearn.tree_.value[index,0,:])) + " ( " +str(np.sum(dt_sklearn.tree_.value[index,0,:])) + " )"
    else:    
        response += "\n"+(2*level+1)*" " + "|-> index " +  str(tree.feature[index]) + " < " + str(tree.threshold[index])
        response += (2*(level+1)+1)*" "+ print_sklearn_tree(tree,tree.children_left[index],level+1)
        response += "\n"+(2*level+1)*" " + "|-> index " +  str(tree.feature[index]) + " >= " + str(tree.threshold[index])
        response += (2*(level+1)+1)*" "+ print_sklearn_tree(tree,tree.children_right[index],level+1)

    return response


dt_sklearn = SklearnDTC(max_depth=3,min_samples_leaf=20,criterion="gini")
dt_sklearn.fit(irisX,irisY)

dt_bts = DecisionTree(gini_impurity,min_leaf_size=20,max_depth=3)
dt_bts.fit(irisX,irisY)

print dt_bts
print "\n" + 50*"-" + "\n"
print print_sklearn_tree(dt_sklearn.tree_)

dt_sklearn = SklearnDTC(max_depth=5,min_samples_leaf=5,criterion="gini")
dt_sklearn.fit(irisX,irisY)

dt_bts = DecisionTree(gini_impurity,min_leaf_size=5,max_depth=5)
dt_bts.fit(irisX,irisY)

print dt_bts
print "\n" + 50*"-" + "\n"
print print_sklearn_tree(dt_sklearn.tree_)

dt_sklearn = SklearnDTR(max_depth=3,min_samples_leaf=20)
dt_sklearn.fit(bostonX,bostonY)

dt_bts = DecisionTree(variance,min_leaf_size=20,max_depth=3)
dt_bts.fit(bostonX,bostonY)

print dt_bts
print "\n" + 50*"-" + "\n"
print print_sklearn_tree(dt_sklearn.tree_)



