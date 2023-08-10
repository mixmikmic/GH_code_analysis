import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


df = pd.read_csv("questions.csv")

print df.columns.values
df1 = df["Question"]


for items in range(10):
    if df1[items] == "x":
        print df["FinalChoice"][items]
    else:
        print df1[items],df["QuestionResponse"][items]
    

allsentences = []

for items in df1:
    sentence = []
    for words in items.split():
        #print words
        pattern = re.search(r'(.*)(\[comma])',words)
        if pattern:
            sentence.append(pattern.group(1).lower())
        else:
            sentence.append(words.lower())
    allsentences.append(sentence)

for items in range(10):
    print allsentences[items]

lista=  []

for items in allsentences:
    captured = " ".join(items)
    lista.append(captured)

listb = []

for items in lista:
    summing = []
    items = items.split()
    for words in items:
        words = words.strip("\',?.")
        #print words
        if words not in stopwords.words("english"):
            summing.append(words)
    listb.append(summing)
    
    

for items in range (10):
    print listb[items]

listc=  []

for items in listb:
    captured = " ".join(items)
    listc.append(captured)

from bs4 import BeautifulSoup
import nltk
from nltk.corpus  import stopwords


total = []

for items in range(len(listc)):
    listc[items] = str(listc[items])
    example1 = BeautifulSoup(listc[items],"lxml")
    letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text() )
    words = letters_only.split()
    for w in words:
        if w not in stopwords.words("english"):
            total.append(w)
    #words = for w in words if not w in stopwords.words("english")]

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

train_data_features = vectorizer.fit_transform(total)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
#print vocab

#Calculate the total count of words
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    if count>50:
        print count,tag
    #print count, tag



