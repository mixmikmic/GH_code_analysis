from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd

# Sample dataset
simple_train = ['Lets play', 'Game time today', 'This game is just awesome!']

#initialize the Vectorizer
vect = CountVectorizer()

# learn the vocab and parse them as features based on the given params.
vect.fit(simple_train)

# get the feature names
vect.get_feature_names()

# convert to a document-term matrix
dtm = vect.transform(simple_train)
dtm

# turn it into an array
dtm.toarray()

# convert the array into a df
df = pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names()) 
df

# check the datatype of the dtm 
type(dtm)

# sparse matrix contains only values where there are non zeros.
print(dtm)

# read the text dataset
path = 'data/sms.tsv'
sms=pd.read_table(path,header=None,names=['label', 'message'])
sms.head()

# check the shape
sms.shape

# diplay based on the categorizations available
sms.label.value_counts()

# convert spams to 1 hams to 0
sms['labels_converted']=sms['label'].apply(lambda x:1 if x=="spam" else 0)

# check if conversion happened.
sms.head()

# get sms message dimensions
X = sms['message']
y = sms['labels_converted']
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

vect = CountVectorizer()

X_train_dtm = vect.fit_transform(X_train)
X_train_dtm

X_train_dtm.toarray().shape

X_test_dtm = vect.transform(X_test)
X_test_dtm

X_test_dtm.toarray().shape

# adding naive bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# fit using Magic Command
get_ipython().magic('time nb.fit(X_train_dtm, y_train)')

y_pred_class = nb.predict(X_test_dtm)
y_pred_class

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred_class)

metrics.confusion_matrix(y_test, y_pred_class)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=1)

clf.fit(X_train_dtm,y_train)

preds = clf.predict(X_test_dtm)
preds

preds.shape

metrics.accuracy_score(y_test,preds)

metrics.confusion_matrix(y_test,preds)

# use a logistic Regression classifier
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()

lreg.fit(X_train_dtm,y_train)

lreg_preds = lreg.predict(X_test_dtm)
lreg_preds

y_pred_prob = lreg.predict_proba(X_test_dtm)
print(y_pred_prob) 
print("-------------")
print(y_pred_prob[:,1])

metrics.accuracy_score(y_test, y_pred_class)



