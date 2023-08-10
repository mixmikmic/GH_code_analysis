import math
from collections import Counter

import numpy as np
import matplotlib.pyplot as mpl
get_ipython().magic('matplotlib inline')

xs = np.arange(0.05, 1.0, 0.05)
ys = -xs * np.log2(xs)
mpl.plot(xs, ys)

def entropy(class_probabilities):
    """Give a list of class probabilities, compute the entropy."""
    return sum(-p * math.log(p, 2)
               for p in class_probabilitiers
               if p) # ignore zero probabilities

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
           for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """Find the entropy from this partition of data into subsets.
    Subsets is a list of lists of labeled data."""
    
    total_count = sum(len(subset) for subset in subsets)
    
    return sum( data_entropy(subset) * len(subset) / total_count
              for subset in subsets )

- 1.0 * np.log2(1.0)

inputs = [
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'},   False),
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yse'},   False),
    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'},   True),
]



