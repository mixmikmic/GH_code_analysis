import pandas as pd
import numpy as np
from collections import defaultdict
import gzip
    
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

train_df = getDF('train.json.gz')
test_df = getDF('test_Helpful.json.gz')

print(train_df.shape)

train_df.isnull().sum()

print('Column price might need to be dropped since', train_df['price'].isnull().sum()/train_df.shape[0] * 100.0, '% of the data is null.')

print('Delete column reviewHash since', train_df['reviewHash'].nunique(), 'unique values, out of', train_df.shape[0], 'exists.')
del train_df['reviewHash']

a = pd.DataFrame.from_dict(dict(train_df['helpful'])).T
train_df1 = pd.concat([train_df, a], axis=1)
train_df1['helpful_rate'] = train_df1['nHelpful']/train_df1['outOf']
train_df1['helpful_rate'].fillna(0, inplace=True)
del train_df1['helpful']
train_df1['reviewTime'] = pd.to_datetime(train_df1['reviewTime'])
del train_df1['unixReviewTime']

from nltk.corpus import stopwords
import string

def reviewText_listed(row):
    all_words = row.split()
    all_words = [w.lower() for w in all_words]
    subset_list = [''.join(c for c in s if c not in string.punctuation) for s in all_words]
    subset_list = [w for w in subset_list if w != '']
    subset_list = [word for word in subset_list if word not in stopwords.words('english')]
    return all_words, len(all_words), subset_list, len(subset_list)

review_allText = {}
review_allText_count = {}
review_keyText = {}
review_keyText_count = {}
vocabulary = []
count = 0
reviewText = list(train_df1['reviewText'])
for text in reviewText:
    all_, all_count, subset_, subset_count = reviewText_listed(text)
    review_allText[count] = all_
    review_allText_count[count] = all_count
    review_keyText[count] = subset_
    review_keyText_count[count] = subset_count
    vocabulary.append(subset_)
    count += 1

from itertools import chain
vocabulary = set(list(chain.from_iterable(vocabulary)))

print('There are', train_df1['reviewerID'].nunique(), 'unique reviewerIDS out of', train_df1.shape[0], 'training records.')
print('There are', train_df1['itemID'].nunique(), 'unique itemIDs out of', train_df1.shape[0], 'training records.')

def RPD(row):
    if (row['max'] - row['min']).days == 0:
        return 0
    else:
        return row['count']/ (row['max'] - row['min']).days

rt_count = train_df1.groupby('reviewerID')['reviewTime'].count()
rt_max = train_df1.groupby('reviewerID')['reviewTime'].max()
rt_min = train_df1.groupby('reviewerID')['reviewTime'].min()

reviewerID_RDP = pd.concat([rt_count, rt_max, rt_min], axis=1, join="inner")
reviewerID_RDP.columns.values[0] = 'count'
reviewerID_RDP.columns.values[1] = 'max'
reviewerID_RDP.columns.values[2] = 'min'
  
reviewerID_RDP['reviewerID_RPD'] = reviewerID_RDP.apply(RPD, axis=1)
reviewerID_RDP.head()

rt_count = train_df1.groupby('itemID')['reviewTime'].count()
rt_max = train_df1.groupby('itemID')['reviewTime'].max()
rt_min = train_df1.groupby('itemID')['reviewTime'].min()

itemID_RPD = pd.concat([rt_count, rt_max, rt_min], axis=1, join="inner")
itemID_RPD.columns.values[0] = 'count'
itemID_RPD.columns.values[1] = 'max'
itemID_RPD.columns.values[2] = 'min'
  
itemID_RPD['itemID_RPD'] = itemID_RPD.apply(RPD, axis=1)
itemID_RPD.head()

train_df1.head(3)

train_df1.shape

review_keyword_length = pd.DataFrame.from_dict(review_keyText_count, orient='index')
review_allword_length = pd.DataFrame.from_dict(review_allText_count, orient='index')

categories = list(train_df1['categories'])
g = list(chain.from_iterable(categories))
print('There are', len(set(list(chain.from_iterable(g)))), 'unique categories values.')

print('There are', train_df1['categoryID'].nunique(), 'unique categoriesID values.')

dummies = pd.get_dummies(train_df1['categoryID']).rename(columns=lambda x: 'categoryID_'+str(x))
# master = pd.concat([train_df1, dummies], axis=1)
master = pd.concat([train_df1, dummies, review_keyword_length, review_allword_length], axis=1)
master.columns.values[17] = 'review_content_len'
master.columns.values[18] = 'review_all_len'
master['review_contentratio'] = master['review_content_len']/master['review_all_len']
del master['categoryID']
master.head(3)

# create a model just using rating, outOf, categoryID_0-5; predict on nHelpful (only if outOf > 0)
# obviously is outOf == 0 then nHelpful == 0, do not need to model this

model = master[master['outOf'] != 0]
print('With all data', master.shape, ', when outOf != 0', model.shape)

columns = ['rating', 'outOf', 'categoryID_0', 'categoryID_1', 'categoryID_2', 'categoryID_3', 'categoryID_4']
X_train = pd.DataFrame(model, columns=columns)
y_train = pd.DataFrame(model.ix[:, 'nHelpful'])                   

from sklearn.model_selection import train_test_split
X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

### Test data
a = pd.DataFrame.from_dict(dict(test_df['helpful'])).T
test_df1 = pd.concat([test_df, a], axis=1)

review1_allText = {}
review1_allText_count = {}
review1_keyText = {}
review1_keyText_count = {}
vocabulary1 = []
count = 0
reviewText = list(test_df1['reviewText'])
for text in reviewText:
    all_, all_count, subset_, subset_count = reviewText_listed(text)
    review1_allText[count] = all_
    review1_allText_count[count] = all_count
    review1_keyText[count] = subset_
    review1_keyText_count[count] = subset_count
    vocabulary1.append(subset_)
    count += 1

review1_keyword_length = pd.DataFrame.from_dict(review1_keyText_count, orient='index')
review1_allword_length = pd.DataFrame.from_dict(review1_allText_count, orient='index')

dummies = pd.get_dummies(test_df1['categoryID']).rename(columns=lambda x: 'categoryID_'+str(x))
master = pd.concat([test_df1, dummies, review1_keyword_length, review1_allword_length], axis=1)
master.columns.values[18] = 'review_content_len'
master.columns.values[19] = 'review_all_len'
master['review_contentratio'] = master['review_content_len']/master['review_all_len']
del master['categoryID']
X_test = pd.DataFrame(master, columns=columns)
X_test.head()

model1 = X_test[X_test['outOf'] != 0]
print('With all data', X_test.shape, ', when outOf != 0', model1.shape)

from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

# kf = StratifiedKFold(y, n_folds=10, random_state=None, shuffle=True)
gridparams = dict(learning_rate=[0.01, 0.1], loss=['ls', 'lad', 'huber', 'quantile'])
# gridparams = dict(learning_rate=[0.01, 0.1, 1, 10], loss=['ls', 'lad', 'huber', 'quantile'])
params = {'n_estimators': 100, 'max_depth': 4}
gbclf = GridSearchCV(ensemble.GradientBoostingRegressor(**params), gridparams, scoring='mean_absolute_error', n_jobs=-1)
# gbclf = GridSearchCV(ensemble.GradientBoostingRegressor(n_estimators= 200, max_depth= 4, criterion= 'mae'), gridparams, scoring='mean_absolute_error', n_jobs=-1)
gbclf.fit(X_train1, y_train1)

print("Best model:")
print(gbclf.best_estimator_)
print("")

y_pred = gbclf.predict(X_valid)
print("Mean absolute error: %0.3f" % mean_absolute_error(np.array(y_valid['nHelpful']), y_pred))

gbreg1 = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='lad', max_depth=4, max_features=None,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=100,
             presort='auto', random_state=None, subsample=1.0, verbose=0,
             warm_start=False)
gbreg1.fit(X_train, y_train)

gbreg1_predictions = []
for i in range(len(X_test)):
    if X_test['outOf'][i] == 0:
        gbreg1_predictions.append(0)
    else:
        gbreg1_predictions.append(round(gbreg1.predict(X_test.ix[i])[0]))

# with rounding
predictions = open("predictions_gbreg_rounding_Helpful.csv", 'w')
predictions.write('userID-itemID-outOf,prediction\n')
for i in range(len(gbreg1_predictions)):
    user = test_user_id[i]
    item = test_item_id[i]
    outof = outOf[i]
    prediction = gbreg1_predictions[i]
    predictions.write(user + '-' + item + '-' + str(outof) + ',' + str(prediction) + '\n')



