import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

X = [{'age': 15.9, 'likes puppies': 'yes', 'location': 'Tokyo'},
     {'age': 21.5, 'likes puppies': 'no',  'location': 'New York'},
     {'age': 31.3, 'likes puppies': 'no',  'location': 'Paris'},
     {'age': 25.1, 'likes puppies': 'yes', 'location': 'New York'},
     {'age': 63.6, 'likes puppies': 'no',  'location': 'Tokyo'},
     {'age': 14.4, 'likes puppies': 'yes', 'location': 'Tokyo'}]

from sklearn.feature_extraction import DictVectorizer
vect = DictVectorizer(sparse=False).fit(X)
vect.transform(X)

vect.get_feature_names()

