import re
import nltk
import pandas as pd
import numpy  as np

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text) # remove html flag, e.g. <br />
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower())+' '.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
# test:
tmp = stream_docs(path='../data/imbd.csv.train')
print next(tmp)
print next(tmp)

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try: 
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
# test:
get_minibatch(stream_docs(path='../data/imbd.csv.train'), size=2)
get_minibatch(stream_docs(path='../data/imbd.csv.train'), size=2)

np.random.seed(0)
df1 = pd.read_csv('../data/imbd.csv.train')
df2 = pd.read_csv('../data/imbd.csv.test')
df  = df2.append(df1, ignore_index=True)
df.reindex(np.random.permutation(df.index))
df.to_csv('../data/imbd.csv', index=False)
df.head()
len(df)

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='../data/imbd.csv')

import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    #print _, X_train
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print 'Accuracy: %.3f'% clf.score(X_test, y_test)

# Update the model with test data
clf.partial_fit(X_test, y_test)

