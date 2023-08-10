get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data/raw/'

np.random.seed(1234)

# load files
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
words = pd.read_csv(os.path.join(DATA_DIR, 'words.csv'), encoding='ISO-8859-1')
users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))

# Number of artists that are present both in training as well as test set.
len(set(train.Artist.unique()) & set(test.Artist.unique()))

# Number of users that are present both in training and test set
len(set(train.User.unique()) & set(test.User.unique()))

print('Number of unique users in the training set: %d'%(len(train.User.unique())))
print('Number of unique users in the test set: %d'%(len(test.User.unique())))

new_users = len(set(test.User.unique()) - set(train.User.unique()))
print('Number of users that are in the test set but not in the training set: %d'%(new_users))

# Lets look at the (artist, user) pair that are in training and test set
def check_membership(artist, user):
    return int(test.loc[(test.Artist == artist) & (test.User == user)].shape[0] != 0)

def count_pairs():
    pairs = 0
    for artist, user in zip(train.Artist, train.User):
        pairs += check_membership(artist, user)
    
    return pairs

common_pairs = count_pairs()
print('Number of (artist, user) pair in training and test set are: %d'%(common_pairs))

artist_user_mean_ratings = train.groupby(['Artist', 'User'])['Rating'].mean().to_dict()

mean_rating = train.Rating.mean() # mean rating irrespective of artist and user information.

features = ['Artist', 'User'] # only consider these two features for now.

X = train[features]
y = train.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1324)

def basic_model(row):
    artist = row['Artist']
    user = row['User']
    
    if (artist, user) in artist_user_mean_ratings:
        return artist_user_mean_ratings[(artist, user)]
    else:
        return mean_rating
    
y_preds = X_test.apply(basic_model, axis=1)

rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print('RMSE on the test set: %f'%(rmse))

