import pandas as pd #To prepare the data
import numpy as np #For the log function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#import the data using pandas
FoR_data = pd.read_csv('fake_or_real_news.csv', usecols=['label', 'text'])
print("There are {} rows in the data set".format(len(movie_data)))
FoR_data.head()

#First randomly select 20% of the data
FoR_sample = FoR_data.sample(1200)
#The training data and the testing data MUST be seperated 
training_frame = FoR_data[~FoR_data.index.isin(FoR_sample.index)]
testing_frame = FoR_sample

def preprocessing(s):
    stop_words = set(stopwords.words("english"))
    s = s.lower()
    s = word_tokenize(s)
    s = [word for word in s if not word in stop_words]
    return s

C = set(training_frame['label']) # FAKE, REAL
D = dict()
for i in range(len(training_frame)):
    D[training_frame.iloc[i,0]] = training_frame.iloc[i,1]

def train_NB(C,D):
    V = set([word for doc in D.keys() for word in preprocessing(doc)])
    N = len(D)
    prior = dict()
    cond_prob = dict()
    N_c = dict()
    T = dict()
    text_c = dict()
    for c in C:
        text_c[c] = []
        for doc in D.items():
            if doc[1] == c:
                for word in doc[0].split():
                    text_c[c].append(word)
        N_c[c] = len([doc for doc in D.items() if doc[1] == c])
        prior[c] = float(N_c[c])/N
        cond_prob[c] = dict()
        T[c] = dict()
        for term in V:
            T[c][term] = text_c[c].count(term)
        for term in V:
            cond_prob[c][term] = float(T[c][term] + 1)/(sum(T[c].values()) + len(V))
 
    return V, prior, cond_prob

def test_NB(C,V,prior,cond_prob,d):
    W = []
    for word in d.split():
        if word in V:
            W.append(word)
    score = dict()
    for c in C:
        score[c] = np.log(prior[c])
        for term in W:
            score[c] += np.log(cond_prob[c][term])
    max_category = sorted(score.items(),key=lambda x: x[1],reverse= True)[0][0]
    return max_category

V, prior, cond_prob = train_NB(C,D)

correct = 0
incorrect = 0
for i in range(len(testing_frame)): 
    if test_NB(C,V,prior,cond_prob,testing_frame.iloc[i,0]) == testing_frame.iloc[i,1]:
        correct += 1
    else:
        incorrect += 1
    print (test_NB(C,V,prior,cond_prob,testing_frame.iloc[i,0]), testing_frame.iloc[i,1], correct/(correct+incorrect))
accuracy = correct/(correct + incorrect)
print("Accuracy = {} %".format(accuracy))

















