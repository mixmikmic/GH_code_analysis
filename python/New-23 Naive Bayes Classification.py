import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.datasets import make_blobs
#100 - total number of data points
#2 - total number of features or number of cols.
#centers (2) - 
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);

y

X

import numpy as np
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()

#Testing with new dataset
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names

categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics','misc.forsale']
#Getting above 5 classes of train data or 5 classes of test data
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

train.target_names

train.data[0]

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#MultinomialNB - More than 2 classes
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
#model = make_pipeline(CountVectorizer(), MultinomialNB())

#model is a pipeline, doing fit on it causes all data to be subjected to tranformation & evaluation
model.fit(train.data, train.target)

labels = model.predict(test.data)

labels

model.predict(['god photos for sale'])

#Finding how good algorithm is - accuracy detection

from sklearn.metrics import confusion_matrix,accuracy_score
mat = confusion_matrix(test.target, labels)

mat

accuracy_score(test.target, labels)

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');

def predict_category(s, model=model):
    pred = model.predict([s])
    print(model.predict_proba([s]))
    return train.target_names[pred[0]]

predict_category('jesus save us')

predict_category("payload to ISS")

predict_category('determine screen resolution')

predict_category('curve of mary cup')

#Understanding tfidf
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop_words - don't consider regular words like is,the etc
tfidf = TfidfVectorizer(stop_words='english')

data = ['hello this is good','good stuff','good place']

tfidf.fit_transform(data).toarray()

# word 'is' is not considered since stop_words is configured 
tfidf.get_feature_names()





