import numpy as np
import pandas as pd

titanic = pd.read_csv("titanictrain.csv")
#titanic_test = pd.read_csv("titanictest.csv")

titanic.drop("PassengerId", axis = 1, inplace = True)
titanic.drop("Name", axis=1, inplace = True)
titanic.drop("Ticket", axis=1, inplace = True)
titanic.drop("Cabin", axis=1, inplace = True)
titanic.drop("Embarked", axis=1, inplace = True)

titanic_test.drop("PassengerId", axis = 1, inplace = True)
titanic_test.drop("Name", axis=1, inplace = True)
titanic_test.drop("Ticket", axis=1, inplace = True)
titanic_test.drop("Cabin", axis=1, inplace = True)
titanic_test.drop("Embarked", axis=1, inplace = True)

titanic_test.drop("Fare", axis=1, inplace = True)
titanic_test.drop("Age", axis=1, inplace = True)

column = ["Pclass", "Sex","SibSp", "Parch", "Survived"]
titanic = titanic.reindex(columns=column)

def f(s):
    if s == "male":
        return 0
    else:
        return 1
titanic["Sex"] =titanic.Sex.apply(f)       #apply rule/function f
titanic.head()

X = titanic 
X = X.values #converting X into np array 

column = ["Pclass", "Sex","SibSp", "Parch"]

def gainRatio(X,column,index):
    y = [X[X[:,index]==k] for k in np.unique(X[:,index])] #split the data based on best filt index/features
    y=np.asarray(y)                                       #saved the multidimensional array as nparray
    n,c = np.unique(X[:,4],return_counts=True)            #for entropy of parent node
    eParent = calcEntro(n,c)
    infoCh =0
    splitInfo=0
    for i in range(y.shape[0]):
        n1 ,c1 = np.unique(y[i][:,4],return_counts=True)   
        temp = calcEntro(n1,c1)                            #entropy of each child node
        xxx = (np.sum(c1)/np.sum(c))
        infoCh = infoCh + (xxx*temp)                    #infoReq for child node
        splitInfo = splitInfo + (-1*xxx)*np.log2(xxx)    #split info for child node 
    iG = eParent - infoCh                               #information gain infoParent - infoChild
   # print("eparent",eParent,"infoG",infoG)
    if(splitInfo == 0):                                 #division by 0 check
    #    print(iG," ",splitInfo)
        return 0 
    gainRatio = (iG/splitInfo)                           #calc Gain Ratio
    return gainRatio

def findBestFit(X,column):
    maxx=-99
    f = -1
    for i in range(len(column)):        #iterate over all the rest columns for best feature
        t = gainRatio(X,column,i)
        if(maxx < t):
          #  print(" ")
            maxx = t
            f = i
    return f,t

def calcEntro(n,c):
    totalSum = np.sum(c)
    ans =0
    for i in range(len(n)):
        m = c[i]/totalSum
        ans = ans + (-1*m)*(np.log2(m))         #entropy formula
    return ans

def printInfo(X,column,level,entropy=0,gainRatio=0,x="NA"):
    if(x=="NA"):
        print("Level",level)
        n,c = np.unique(X[:,4],return_counts=True)
     #   print(len(n))
        for i in range(len(n)):
                print("Count of",n[i],c[i])
        print("Entropy",entropy)
        print("Reached leaf Node")
    else:
        print("Level",level)
        n,c = np.unique(X[:,4],return_counts=True)
    # print(len(n))
        for i in range(len(n)):
                print("Count of",n[i],c[i])
        print("Entropy",entropy)
        print("splitting Feature based on",x," with gain ratio",gainRatio)

def decisionTree(X,column,level=0):
    if(len(np.unique(X[:,4]))==1):    #base case if node is pure 
        printInfo(X,column,level)
        print(" ")
        return
    if(len(column)==0):             #base case if no feature left to split
        n,c = np.unique(X[:,4],return_counts=True)
        entropy = calcEntro(n,c)
        printInfo(X,column,level,entropy)
        print(" ")
        return
    index,gainRatio= findBestFit(X,column)   #find the best fit feature and its gainRatio
    n,c = np.unique(X[:,4],return_counts=True)
    entropy = calcEntro(n,c)                        
    name = column[index]
    printInfo(X,column,level,entropy,gainRatio,name)
    print(" ")
    level += 1                      #inc the level
    y = [X[X[:,index]==k] for k in np.unique(X[:,index])]     #split the data on basis of feature
    y1=np.asarray(y)
    del column[index]
    
    for i in range(y1.shape[0]):
        #y1 = np.delete(y1[i], np.s_[index:index+1], axis=1)
        decisionTree(y1[i],column,level)
    return

column = ["Pclass", "Sex","SibSp", "Parch"]
decisionTree(X,column,0)

column

