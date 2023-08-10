import pandas as pd 
import numpy as np 
from Madefunctions import *
import math 

#Titanic Dataset 
titanic = pd.read_csv("titanic.csv")
titanic.head()

newdf = preprocess(titanic)

newdf.head()

categorical = splitter_cat(newdf)
categorical.head()

categorical = removeUcol(categorical)

train, test = traintestsplit(categorical)

train.head()

train2 = train.drop("name", axis=1)

train3 = train2.drop("ticket",axis=1)

train3.head()

test.head()

test2 = test.drop("name", axis=1)

test3 = test2.drop("ticket", axis=1)

test3.head()

prob, col, L = probdict(train3)

prob

test3.head()

col

L

prob[1]["embarked"]["C"]

T =  predict(prob, test3, col,L)

test3.head()

naiveaccuracy(T)

d = {}
f = {}
# ls = [2,2,5]
# ls2 = [5,7,8]
features = train.columns.values
features = features.tolist()
for feature in train:
    d[i] = []
    f[feature] = []
    
print(d)
print(f)

ls = list(set(train3["survived"]))
d = {}
f = {}

features = train3.columns.values
features = features.tolist()
for i in ls:
    d[i] = []
    for feature in features:
        f.update({feature:[{1:"a"},{2:"b"}]})
    d[i] = f
    
    
print(d)

def binning(df):
    col = input("Pick a column to bin: ")
    colmin = min(df[col])
    print(colmin)
    colmax = max(df[col])
    print(colmax)
    diff = colmax - colmin 
    print(diff)
    numb = int(input("number of bins: "))
    bins = diff / numb
    ls = []
    
    for n in range(1,numb+1):
        bb = str("b") + str(n)
#         ls.append(bb)
        bb = colmin + (bins*n) 
        ls.append(int(bb))
#         print(bb)
    print(ls)
    t = str("0-")
    for row in df[col]:
        
        for i in range(0,len(ls)):
            
            if row < ls[i]:
                s = t + str(ls[i])
                df[col] = df[col].replace(row, s)
                t = str(ls[i]) + "-"
                
        
        
    print(df.head())  

binning(clean)



