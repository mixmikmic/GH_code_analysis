import numpy as np
import re

from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2:])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        try:
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
        except StopIteration:
            return None, None
    return docs, y
        
next(stream_docs(path='./movie_review.csv'))

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# note that, by choosing a large number of features in the HashingVectorizer,
# we reduce the chance to cause hash collisions but we also increase the number
# of coefficients in our logistic regression model.
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_review.csv')

# out of 50k docs we've, let's train 45k docs.
import pyprind
pbar = pyprind.ProgBar(45) # 45 iterations of 1000 docs each
classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, 1000) # 45 iterations of 1000 docs each
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
    

# use the remaining 5k to compute score
X_test, y_test = get_minibatch(doc_stream, 5000)
X_test = vect.transform(X_test)
print "Score: %.3f" %(clf.score(X_test, y_test))

