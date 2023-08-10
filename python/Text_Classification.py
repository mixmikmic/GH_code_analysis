import numpy as np
import pandas as pd
import os
import time

start_time = time.time()
start_time

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words

# Our own list of some block words to be avoided  
block_words = ['newsgroups', 'xref', 'path', 'from', 'subject', 'sender', 'organisation', 'apr','gmt', 'last','better','never','every','even','two','good','used','first','need','going','must','really','might','well','without','made','give','look','try','far','less','seem','new','make','many','way','since','using','take','help','thanks','send','free','may','see','much','want','find','would','one','like','get','use','also','could','say','us','go','please','said','set','got','sure','come','lot','seems','able','anything','put', '--', '|>', '>>', '93', 'xref', 'cantaloupe.srv.cs.cmu.edu', '20', '16', "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", '21', '19', '10', '17', '24', 'reply-to:', 'thu', 'nntp-posting-host:', 're:','25''18'"i'd"'>i''22''fri,''23''>the','references:','xref:','sender:','writes:','1993','organization:']

import urllib.request
urllib.request.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz", "a.tar.gz")

import tarfile
tar = tarfile.open("a.tar.gz")
tar.extractall()
tar.close()

##Make a list of the folders in the dataset
directory = [f for f in os.listdir('./20_newsgroups') if not f.startswith('.')]
directory

vocab = {}
for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        ##Path of each file 
        path = './20_newsgroups/' + directory[i] + '/' + files[j]
        
        ##open the file and read it
        text = open(path, 'r', errors='ignore').read()
        
        for word in text.split():
            
            ## If word doesnt contain any special character then create the dictionary
            if len(word) != 1:  
                
                ##Check if word is a non stop word or non block word(we have created) only then proceed
                if not word.lower() in stop_words:
                    if not word.lower() in block_words:     
                        ##If word is already in dictionary then we just increment its frequency by 1
                        if vocab.get(word.lower()) != None:
                            vocab[word.lower()] += 1

                        ##If word is not in dictionary then we put that word in our dictinary by making its frequnecy 1
                        else:
                            vocab[word.lower()] = 1
            
# vocab

import operator
sorted_vocab = sorted(vocab.items(), key= operator.itemgetter(1), reverse= True)
# sorted_vocab

# Dictionary containing the most occuring k-words.
kvocab={}

# Frequency of 1000th most occured word
z = sorted_vocab[2000][1]

for x in sorted_vocab:
    kvocab[x[0]] = x[1]
    
    if x[1] <= z:
        break

len(kvocab)

sorted_vocab[0:100]

features_list = list(kvocab.keys())

## Create a Dataframe containing features_list as columns 
df = pd.DataFrame(columns = features_list)


## Filling the x_train values in dataframe 

for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        ##Insert a row at the end of Dataframe with all zeros
        df.loc[len(df)] = np.zeros(len(features_list))
        
        ##Path of each file 
        path = './20_newsgroups/' + directory[i] + '/' + files[j]
        
        ##open the file and read it
        text = open(path, 'r', errors='ignore').read()
        
        
        for word in text.split():
            if word.lower() in features_list:
                df[word.lower()][len(df)-1] += 1
                

# df.head()

# df.describe()

## Making the 2d array of x
x = df.values
x.shape

## Creating  y array containing labels for classification 

y = []

for i in range(len(directory)):
    ##Create a list of files in the given dictionary 
    files = os.listdir('./20_newsgroups/' + directory[i])
 
    for j in range(len(files)):
        y.append(i)

y = np.array(y)
y.shape

from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

train_score, test_score

end_time = time.time()
total_time = end_time - start_time
total_time

