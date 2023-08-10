import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

import os
print("file size: %d GB" % (os.path.getsize("data/movies.txt") / 1024 ** 3))

with open("data/movies.txt") as f:
    print(f.read(4000))

def review_iter(f):
    current_post = []
    for line in f.readlines():
        if line.startswith("product/productId"):
            if len(current_post):
                score = current_post[3].strip("review/score: ").strip()
                review = "".join(current_post[6:]).strip("review/text: ").strip()
                yield int(float(score)), review
            current_post = []
        else:
            current_post.append(line)

n_reviews = 0
with open("data/movies.txt") as f:
    for r in review_iter(f):
        n_reviews += 1
print("Number of reviews: %d" % n_reviews)

from itertools import islice

with open("data/movies.txt") as f:
    reviews = islice(review_iter(f), 10000)
    scores, texts = zip(*reviews)
print(np.bincount(scores))

from itertools import izip_longest
# from the itertools recipes
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def preprocess_batch(reviews):
    reviews_filtered = [r for r in reviews if r is not None and r[0] != 3]
    scores, texts = zip(*reviews_filtered)
    polarity = np.array(scores) > 3
    return polarity, texts

from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(decode_error="ignore")

with open("data/movies.txt") as f:
    reviews = islice(review_iter(f), 10000)
    polarity_test, texts_test = preprocess_batch(reviews)
    X_test = vectorizer.transform(texts_test)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=0)

accuracies = []
with open("data/movies.txt") as f:
    training_set = islice(review_iter(f), 10000, None)
    batch_iter = grouper(training_set, 10000)
    for batch in batch_iter:
        polarity, texts = preprocess_batch(batch)
        X = vectorizer.transform(texts)
        sgd.partial_fit(X, polarity, classes=[0, 1])
        accuracies.append(sgd.score(X_test, polarity_test))

plt.plot(accuracies)



