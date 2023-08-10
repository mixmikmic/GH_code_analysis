import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

df = pd.read_csv('../datasets/titanic_train.csv')

df.head(2)

df.dtypes

df.isnull().sum()

for item in df:
    print item, df[item].nunique()

# I'll create some dummy-coded columns for which letter was in the cabin and the
# number of the cabin. In the case there are multiple, just using first.
# if null, just coding 0s!
# (This is intended to be kind of dumb to see if features will be eliminated!)

cabin_letter = df.Cabin.map(lambda x: 'Z' if pd.isnull(x) else x.split()[0][0])
cabin_letter.unique()

cabin_dummy = pd.get_dummies(cabin_letter, prefix='cabin')
cabin_dummy.head()

cabin_dummy.drop('cabin_Z', axis=1, inplace=True)

def cabin_numberer(x):
    try:
        return int(x.split()[0][1:])
    except:
        return 0

cabin_num = df.Cabin.map(cabin_numberer)
cabin_num.unique()

df['cabin_number'] = cabin_num
df = pd.concat([df, cabin_dummy], axis=1)

# Lets be real: a Persons name, their passenger ID, Ticket number 
# aren't going to be useful features.

# Keep passengerid in for the sake of example
# Remove name, ticket, and cabin
df.drop('PassengerId', inplace=True, axis=1)
df.drop('Name', inplace=True, axis=1)
df.drop('Ticket', inplace=True, axis=1)
df.drop('Cabin', inplace=True, axis=1)

# impute the median for Age to fill the nulls
df.Age.fillna(df.Age.median(), inplace=True)

# Mean and median age values are very close (28 and 29) 
# we can assume our distribution of age is fairly normal

# make dummy variables for embarked, dropping the original Embarked column 
# and 'S' (the most common embarcation point)
df = pd.concat([df, pd.get_dummies(df.Embarked)], axis=1)
df.drop('S', inplace=True, axis=1)
df.drop('Embarked', inplace=True, axis=1)

# I could just use drop_first = True, but there is more than one way to do anything.

# instead of sex, create a column called 'male' with a binary value
df['Male'] = df.Sex.apply(lambda x: 'female' not in str(x))

# drop the original Sex column
df.drop('Sex', inplace=True, axis=1)

df.head()
# Data After cleaning and parsing

# this list of column names will come in handly later.
cols = list(df.columns)
cols.remove('Survived')

X = df[cols]
y = df.Survived.values

from sklearn.feature_selection import SelectKBest, chi2, f_classif

# build the selector (we'll build one with each score type)
skb_f = SelectKBest(f_classif, k=5)
skb_chi2 = SelectKBest(chi2, k=5)

# train the selector on our data
skb_f.fit(X, y)
skb_chi2.fit(X, y)

# examine results
kbest = pd.DataFrame([cols, list(skb_f.scores_), list(skb_chi2.scores_)], 
                     index=['feature','f_classif','chi2 score']).T.sort_values('f_classif', ascending=False)
kbest

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
selector = RFECV(lr, step=1, cv=10)
selector = selector.fit(X, y)

print selector.support_
print selector.ranking_

 

# the column names correspond to the one below.  RFECV only excluded a few features.
rfecv_columns = np.array(cols)[selector.support_]
rfecv_columns

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
Xs = ss.fit_transform(X)

from sklearn.linear_model import LogisticRegressionCV

lrcv = LogisticRegressionCV(penalty='l1', Cs=100, cv=10, solver='liblinear')
lrcv.fit(Xs, y)

lrcv.C_

# What are the best coefficients according to a model using lasso?
coeffs = pd.DataFrame(lrcv.coef_, columns=X.columns)
coeffs_t = coeffs.transpose()
coeffs_t.columns = ['lasso_coefs']
coeffs_abs = coeffs_t.abs().sort_values('lasso_coefs', ascending=False)
coeffs_abs

# A few variables were eliminated. Not totally consistent with RFECV - 
# More features were eliminated by the Lasso method

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

kbest_columns = kbest.feature.values[0:5]
lasso_columns = coeffs_abs.index[coeffs_t.lasso_coefs != 0]

lr = LogisticRegression(C=lrcv.C_[0], penalty='l1', solver='liblinear')

# defining a function to test our best features head-to-head
def score(X):
    scores = cross_val_score(lr, X, y, cv=5)
    return scores.mean(), scores.std()

# list of all our lists of best features being executed in the score function
all_scores = [
    score(X[kbest_columns]),
    score(X[rfecv_columns]),
    score(X[lasso_columns]),
    score(X)]

#putting results into a dataframe
pd.DataFrame(all_scores, columns=['mean score', 'std score'], index = ['kbest', 'rfecv', 'lr', 'all'])

# There is very, very little difference in performance 
# of different features for this dataset.

coeffs_t.sort_values('lasso_coefs').plot(kind='bar')

