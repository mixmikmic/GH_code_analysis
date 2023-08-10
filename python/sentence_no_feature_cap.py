import re, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from spacy.en import English ##Note you'll need to install Spacy and download its dependencies
parser = English()
import string

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]

import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('C:/Users/Administrator/Documents/Github/mcnulty_yelp/data/bigtext.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
#     mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
#     text = mentionFinder.sub("@MENTION", text)
    text = re.sub('[^a-zA-Z0-9 ]','',text)
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()
#     text = correct(text)
    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(str(TextBlob(sample).correct()))

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

import pandas as pd
df = pd.read_csv('C:/Users/Administrator/Documents/Github/mcnulty_yelp/data/sentence_raw.csv',encoding = "ISO-8859-1")

df2 = df[['sentence','category']]

##Collapse some
df2[df2['category']=="wait"] = "service" 
df2[df2['category']=="value"] = "overall"

df2.category.value_counts()/df2.shape[0]

df2.sentence = df2.sentence.apply(cleanText)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

hashvectorizer = HashingVectorizer(ngram_range=(1,3),tokenizer=tokenizeText)
vectorizer = CountVectorizer(ngram_range=(1,3),min_df=3,tokenizer=tokenizeText)
tfvectorizer = TfidfVectorizer(ngram_range=(1,3),min_df = 3,tokenizer=tokenizeText)

## Gets the count of each word in each sentence (Count Vectorizer)
countfeature = vectorizer.fit_transform(df2.sentence)
# lsa_count = TruncatedSVD(n_components=50,n_iter=100).fit_transform(countfeature)
# lsa_count_df = pd.DataFrame(lsa_count)
featuredf_count = pd.DataFrame(countfeature.A, columns=vectorizer.get_feature_names())
df3_count = pd.concat((df2,featuredf_count),axis=1)
# df3_lsa_count = pd.concat((df2,lsa_count_df),axis=1)
print(df3_count.info())
# print(df3_lsa_count.info())
df3_count.to_pickle("df3_count_no_cap.pkl")
# df3_lsa_count.to_pickle("df3_lsa_count.pkl")

from sklearn.feature_extraction.text import TfidfTransformer
tffeature = TfidfTransformer().fit_transform(countfeature)
featuredf_tf = pd.DataFrame(tffeature.A, columns=vectorizer.get_feature_names())
df3_tf = pd.concat((df2,featuredf_tf),axis=1)
featuredf_tf.info()
df3_tf.to_pickle("df3_tf_no_cap.pkl")

df3_count = pd.read_pickle("df3_count_no_cap.pkl")
df3_tf = pd.read_pickle("df3_tf_no_cap.pkl")

df3_count.iloc[:,1].value_counts()/df3_count.shape[0]

expanded_df = pd.concat((df3_count,df3_count[df3_count.iloc[:,1]=='ambiance']),axis=0)

expanded_df.iloc[:,1].value_counts()/expanded_df.shape[0]

from sklearn.cross_validation import train_test_split 

##Split into train and test at 75/25
train, test = train_test_split(df3_count.values,test_size = 0.25, random_state=1)

##Split X & Y
X_train = train[:,2:]
Y_train = train[:,1]
X_test = test[:,2:]
Y_test = test[:,1]

### Temp just to check oversampled data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

## NaiveBayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(X_train,Y_train)
nb_Y_pred = nb.predict(X_test)
print("NB Accuracy: ",np.mean(nb_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,nb_Y_pred))

## Logistic OVR
## C=1
from sklearn.linear_model import LogisticRegression
log = LogisticRegression().fit(X_train,Y_train)
log_Y_pred = log.predict(X_test)
print("Logistc Accuracy: ",np.mean(log_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,log_Y_pred))

## Logistic OVR
## C=1
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(class_weight='balanced').fit(X_train,Y_train)
log_Y_pred = log.predict(X_test)
print("Logistc Accuracy: ",np.mean(log_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,log_Y_pred))

from sklearn.cross_validation import cross_val_score

print(cross_val_score(nb,df3_count.iloc[:,2:].values,df3_count.iloc[:,1].values,cv=10))
print(cross_val_score(log,df3_count.iloc[:,2:].values,df3_count.iloc[:,1].values,cv=10))

##Other Models, slow and generally less accurate based on previous tests

## Linear SVC
from sklearn.svm import SVC
svcl = SVC(kernel='linear').fit(X_train,Y_train)
svcl_Y_pred = svcl.predict(X_test)
print("SVC Linear Accuracy: ",np.mean(svcl_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,svcl_Y_pred))

## RBF SVC
# from sklearn.linear_model import SGDClassifier
# sgd = SGDClassifier(loss='perceptron',penalty='elasticnet',l1_ratio=0.5).fit(X_train,Y_train)
# sgd_Y_pred = sgd.predict(X_test)
# print("SGD Perceptron Accuracy: ",np.mean(sgd_Y_pred == np.array(Y_test)))
# print(classification_report(Y_test,sgd_Y_pred))

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier().fit(X_train,Y_train)
dt_Y_pred = dt.predict(X_test)
print("Decision Tree Accuracy: ",np.mean(dt_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,dt_Y_pred))

## Random Forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train,Y_train)
rf_Y_pred = rf.predict(X_test)
print("Random Forests Accuracy: ",np.mean(rf_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,rf_Y_pred))

##Split into train and test at 75/25
train, test = train_test_split(df3_tf.values,test_size = 0.25, random_state=1)

##Split X & Y
X_train = train[:,2:]
Y_train = train[:,1]
X_test = test[:,2:]
Y_test = test[:,1]

df3_tf.info()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

## NaiveBayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(X_train,Y_train)
nb_Y_pred = nb.predict(X_test)
print("NB Accuracy: ",np.mean(nb_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,nb_Y_pred))

## Logistic OVR
## C = 1
from sklearn.linear_model import LogisticRegression
log = LogisticRegression().fit(X_train,Y_train)
log_Y_pred = log.predict(X_test)
print("Logistc Accuracy: ",np.mean(log_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,log_Y_pred))

## Logistic OVR
## C = 1
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(class_weight = 'balanced').fit(X_train,Y_train)
log_Y_pred = log.predict(X_test)
print("Logistc Accuracy: ",np.mean(log_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,log_Y_pred))

##Other Models

##KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
knn_Y_pred = knn.predict(X_test)
print("KNN Accuracy: ",np.mean(knn_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,knn_Y_pred))

## Linear SVC
from sklearn.svm import SVC
svcl = SVC(kernel='linear').fit(X_train,Y_train)
svcl_Y_pred = svcl.predict(X_test)
print("SVC Linear Accuracy: ",np.mean(svcl_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,svcl_Y_pred))

## RBF SVC
# from sklearn.linear_model import SGDClassifier
# sgd = SGDClassifier(loss='perceptron',penalty='elasticnet',l1_ratio=0.5).fit(X_train,Y_train)
# sgd_Y_pred = sgd.predict(X_test)
# print("SGD Perceptron Accuracy: ",np.mean(sgd_Y_pred == np.array(Y_test)))
# print(classification_report(Y_test,sgd_Y_pred))

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier().fit(X_train,Y_train)
dt_Y_pred = dt.predict(X_test)
print("Decision Tree Accuracy: ",np.mean(dt_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,dt_Y_pred))

## Random Forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train,Y_train)
rf_Y_pred = rf.predict(X_test)
print("Random Forests Accuracy: ",np.mean(rf_Y_pred == np.array(Y_test)))
print(classification_report(Y_test,rf_Y_pred))

##Pickle models
from sklearn.externals import joblib
joblib.dump(nb,'nb_sentence.pkl')
joblib.dump(knn,'knn_sentence.pkl')
joblib.dump(log,'log_sentence.pkl')
joblib.dump(svcl,'svcl_sentence.pkl')
joblib.dump(sgd,'sgd_sentence.pkl')
joblib.dump(dt,'dt_sentence.pkl')
joblib.dump(rf,'rf_sentence.pkl')

new_text = df3_count.sentence
vocab_list = np.array(df3_count.columns)[2:]

vocab_dict = dict.fromkeys(vocab_list)

countvectorizer2 = CountVectorizer(ngram_range=(1,3),min_df = 3,tokenizer=tokenizeText,vocabulary=vocab_list)

check = countvectorizer2.transform(new_text[0:100].reset_index(drop=True))
check

large_df = pd.read_pickle("C:/Users/kenndanielso/Documents/Github/mcnulty_yelp/data/fina_df.pkl")

large_df.info()

countvectorizer2 = CountVectorizer(ngram_range=(1,3),min_df = 3,tokenizer=tokenizeText,vocabulary=vocab_list)
large_df_feature = countvectorizer2.transform(large_df.sentence)

large_df_feature_df = pd.DataFrame(large_df_feature.A, columns=countvectorizer2.get_feature_names())
large_df = pd.concat((large_df,large_df_feature_df),axis=1)







##Adds sentiment as a feature. Note that I added 1 because some algorithms won't accept negative sentiment scores
##Sentiment scores is based on TextBlob where it goes from -1.0 to 1.0 (negative to positive)
##Multiplied by 2.5 to scale to 5-star ratings

large_df['senti_score'] = large_df['sentence'].apply(lambda x: (TextBlob(x).sentiment[0] + 1)*5/2)
large_df['senti_subj'] = large_df['sentence'].apply(lambda x: TextBlob(x).sentiment[1])

large_df.to_pickle("final_sentence_df.pkl")



