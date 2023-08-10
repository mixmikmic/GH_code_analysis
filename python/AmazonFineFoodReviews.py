get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3
import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

con=sqlite3.connect('database.sqlite')
messages=pd.read_sql_query(""" SELECT Score, Summary from Reviews where Score!=3""", con)
messages.head()

def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

Score=messages['Score']
Score= Score.map(partition)
Summary=messages['Summary']
X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2, random_state=42)

tmp=messages
tmp['Score']=tmp['Score'].map(partition)
tmp.head()

stemmer=PorterStemmer()
def stem_tokens(token, stemmer):
    stemmed=[]
    for item in token:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    token=nltk.word_tokenize(text)
    stems=stem_tokens(token,stemmer)
    return ' '.join(stems)

intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)

corpus=[]
for text in X_train:
    text=text.lower()
    text=text.translate(trantab)
    text=tokenize(text)
    corpus.append(text)
    
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)        
        
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

test_set=[]
for text in X_test:
    text=text.lower()
    text=text.translate(trantab)
    text=tokenize(text)
    test_set.append(text)
    
X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

df=pd.DataFrame({'Before': X_train, 'After': corpus})
df.head()

predictors={}
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train_tfidf, y_train)
predictors['Multinomial']= model.predict(X_test_tfidf)

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB().fit(X_train_tfidf, y_train)
predictors['Bernoulli']= model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e6)
logreg.fit(X_train_tfidf, y_train)
predictors['Logistic'] = logreg.predict(X_test_tfidf)

def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in predictors.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictors['Logistic']))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(Score)))
    plt.xticks(tick_marks, set(Score), rotation=45)
    plt.yticks(tick_marks, set(Score))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = confusion_matrix(y_test, predictors['Logistic'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()



