from nltk.corpus import stopwords
import os
import re
import pandas as pd
import numpy as np

stop_words = set(stopwords.words('english'))
block_wrds = ['last','better','never','every','even','two','good','used','first','need','going','must','really','might','well','without','made','give','look','try','far','less','seem','new','make','many','way','since','using','take','help','thanks','send','free','may','see','much','want','find','would','one','like','get','use','also','could','say','us','go','please','said','set','got','sure','come','lot','seems','able','anything','put']
not "like" in block_wrds

dictionary = {}
type(dictionary)
count=0
for file in os.listdir("20_newsgroups"):
    for files in os.listdir("20_newsgroups/"+file):
        #print(file,files)
        f = open("20_newsgroups/"+file+"/"+files,'r',errors='ignore')
        message = f.read()
        message = message.split()
        k =1
        for i in message:
            count +=1
            if(i.isalpha() == True and len(i) > 1):
                if not i.lower() in stop_words:
                    if not i.lower() in block_wrds:
                        if(i.lower() in dictionary.keys()):
                            dictionary[i.lower()] = dictionary[i.lower()] +1
                        else:
                            dictionary[i.lower()] = 1


        f.close()

dictionary

import operator
sorted_vocab = sorted(dictionary.items(), key= operator.itemgetter(1), reverse= True)
sorted_vocab

top_val = []
sorted_vocab[1000][1]
size = len(sorted_vocab)
for i in range(size):
    if(sorted_vocab[1000][1] <= sorted_vocab[i][1]):
        top_val.append(sorted_vocab[i][0])

top_val[0:100]

df = pd.DataFrame(columns = top_val)
df.columns

df = pd.DataFrame(columns = top_val)
df.columns
count=0
for file in os.listdir("20_newsgroups"):
    for files in os.listdir("20_newsgroups/"+file):
        count=count+1
        #print(file,files)
        df.loc[len(df)] = np.zeros(len(top_val))
        f = open("20_newsgroups/"+file+"/"+files,'r',errors='ignore')
        message = f.read()
        message = message.split()
        k =0
        for i in message:
            if(i.lower() in df.columns):
                df[i.lower()][len(df)-1] += 1
        f.close()
count

df.shape

y=[]
i=0
count=0
for file in os.listdir("20_newsgroups"):
    for files in os.listdir("20_newsgroups/"+file):
        #print(file,files)
        count+=1
        y.append(i)
    i=i+1

y = np.array(y)
y.shape,df.shape
x = df.values
count

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.30,shuffle=True, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

train_score, test_score

newData = df
newData['out'] = y
newData.to_csv("textClassification.csv")

data = pd.read_csv("textClassification.csv")
Y = data["out"]
print(data.shape)
data.drop(['out'], axis = 1, inplace = True)
data.drop(['Unnamed: 0'], axis = 1, inplace = True)

print(data.shape)

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, Y, test_size = 0.30,shuffle=True, random_state = 0)

f_list = list(data)
xx = X_train[Y_train == 0]
xx.shape,Y_train.shape,X_train.shape,X_test.shape

g = set(Y_train)
d = X_train[Y_train == 1]
d.shape

def train(X_train,Y_train):
    result = {}
    set_class = set(Y_train)
    result["total_data"] = len(Y_train)
    for curr_class in set_class:
        result[curr_class] = {}
        #all the x_train rows whose Y is curr_class
        curr_class_rows = (Y_train == curr_class)
        X_train_curr = X_train[curr_class_rows]
        Y_train_curr = Y_train[curr_class_rows]
        #traverse through all the features of X_train and get the sum of each word and save it in the dict
        #i.e result[class][word] = count of word appearance in the doc
        sums = 0
        for x in f_list:
            result[curr_class][x] = X_train_curr[x].sum()
            sums = sums+result[curr_class][x]
        result[curr_class]["total_count"] = sums
    return result

dictionary = train(X_train,Y_train)

len(dictionary[0]),len(f_list)

dictionary[4]['total_count']

def probablity(dictionary,x,current_class):
    output= np.log(dictionary[current_class]["total_count"])-np.log(dictionary["total_data"])
   # num_features = len(dictionary[current_class].keys())-1;
    for j in f_list:
        count_current_class_with_word_i = dictionary[current_class][j] + 1 
        count_current_class = dictionary[current_class]["total_count"] + len(f_list)
        current_xj_prob = np.log(count_current_class_with_word_i) -np.log(count_current_class)
        output = output + current_xj_prob
    return output 

def predictSinglePoint(dictionary,x):
    classes = dictionary.keys()
    best_p = -1000
    best_class = -1
    first_run = True
    for current_class in classes:
        if(current_class == "total_data"):
            continue
        p_curr_class = probablity(dictionary,x,current_class)
        if(first_run or p_curr_class > best_p):
            best_p = p_curr_class
            best_class = current_class
        first_run = False
    return best_class

def predict(dictionary,X_test):
    Y_pred = []
    for j in range(len(X_test)):
        
        x_class = predictSinglePoint(dictionary,j)
        Y_pred.append(x_class)
    return Y_pred

d = ()
for f in X_test.iterrows():
    d = f

for i in d[1]:
    print(i)

Y_pred = predict(dictionary,X_test)

