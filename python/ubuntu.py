import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt

# use tfidf for base line
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_data = pd.read_csv('../../data/udc/train.csv')

train_data.Label = train_data.Label.astype('category')
train_data.describe()

train_data.head()

train_data.Label.hist()
plt.show()

train_data_context_len = train_data.Context.str.split(" ").apply(len)
train_data_context_len.hist(bins=40)
plt.show()
train_data_context_len.describe()

train_data_utterance_len = train_data.Utterance.str.split(" ").apply(len)
train_data_utterance_len.hist(bins=40)
plt.show()
train_data_utterance_len.describe()

test_data = pd.read_csv('../../data/udc/test.csv')
test_data.head()

validation_data = pd.read_csv('../../data/udc/valid.csv')

def recallk(preds, k=1):
    total = len(preds)
    correct = 0
    for pred in preds:
        # the first one is the ground truth, its index is always 0
        if 0 in pred[:k]:
            correct += 1
    return correct / total

def predict_random():
    return np.random.permutation(np.arange(10))

random_predictions = [predict_random() for _ in range(len(test_data))]

for k in [1, 2, 3, 10]:
    print('Recall k={} accuracy: {}'.format(k, recallk(random_predictions, k)))

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(np.append(train_data.Context.values, train_data.Utterance.values))

tfidf_vectorizer.transform(["this is ubuntu"])

def predict_tfidf(context, utterances, debug=False):
    # vectorize, NOTE: wrap context with [] instead of just a string, sklearn expect iterable string
    # [1 x n] where n is size of dictionary
    vec_context = tfidf_vectorizer.transform([context])
    # [10 x n] 10 beause we have 1 ground truth and 9 ex
    vec_doc = tfidf_vectorizer.transform(utterances)
    if debug:
        print(vec_context.shape, vec_doc.shape)
    # use dot product to measure similarity of the resulting vectors
    result = np.dot(vec_doc, vec_context.T).todense()
    result = np.asarray(result).flatten()
    return np.argsort(result, axis=0)[::-1]

predict_tfidf(test_data.Context[0], test_data.iloc[0, 1:].values, True)

tfidf_predictions = []
for i in range(len(test_data)):
    tfidf_predictions.append(predict_tfidf(test_data.Context[i], test_data.iloc[i, 1:].values))

for k in [1, 2, 3, 10]:
    print('Recall k={} accuracy: {}'.format(k, recallk(tfidf_predictions, k)))



