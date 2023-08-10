import numpy as np
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

def find_files(path):
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            yield os.path.join(root, filename)

newsgroup_files = sorted(find_files('./resources/20news-18828/'))

assert len(newsgroup_files) != 0,        "The files of the resource were not found"

# `labels` holds the name of the labels and `target` 
# holds the representation of the labels using integers
labels, target = np.unique(
    [os.path.basename(os.path.dirname(fname))
        for fname in newsgroup_files],
    return_inverse=True
)

vectorizer = TfidfVectorizer(input='filename',
                             decode_error='replace',
                             stop_words='english',
                             max_features=10000)
document_matrix = vectorizer    .fit_transform(newsgroup_files)    .toarray() # The vectorizer returns a sparse matrix
               # we turn into a dense one

sss = StratifiedShuffleSplit(2, test_size=0.2, random_state=0)
train_idx, test_idx = next(sss.split(document_matrix, target))

train_data = document_matrix[train_idx]
train_target = target[train_idx]

test_data = document_matrix[test_idx]
test_target = target[test_idx]

np.savez_compressed('./resources/newsgroup.npz',
                    train_data=train_data, train_target=train_target,
                    test_data=test_data, test_target=test_target,
                    labels=labels)

