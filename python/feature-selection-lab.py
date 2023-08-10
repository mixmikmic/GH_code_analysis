import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

df = pd.read_csv('./datasets/titanic_train.csv')

# A:
df.head()

df.info() 

# Lets just drop obviously unuseful features
drop_df = df.drop(labels=['PassengerId','Ticket','Name'], axis=1)

# Maybe we can use the first letter to engineer this feature
drop_df['Cabin'].unique()

# Z cabin for all the nan values
cabin_dummy = pd.get_dummies(drop_df['Cabin'].map(lambda x: 'Z' if pd.isnull(x) else x.split()[0][0]), prefix='Cabin')
cabin_dummy = cabin_dummy.astype('int64')

# We will use Z as reference row
cabin_dummy.drop(labels='Cabin_Z', axis=1, inplace=True)

# combine with main df
drop_df = pd.concat([drop_df,cabin_dummy],axis=1)
drop_df.head()

# Now we will get room numbers
# Nan values default to 0
drop_df['Cabin_num'] = df['Cabin'].map(lambda x: 0 if pd.isnull(x) or x.split()[0][1:] == '' else int(x.split()[0][1:]))
drop_df.head()

# Only 2 nan values
drop_df[drop_df['Embarked'].isnull()].index

# Create dummy variables for embark column
embark_dummy = pd.get_dummies(drop_df.Embarked,'Embark')
embark_dummy = embark_dummy.astype('int64')

# Combine with main frame
drop_df = pd.concat([drop_df, embark_dummy], axis=1)
drop_df.head()

# Clean age column
# wants to replace with median age but lets check statistics first
print(drop_df.Age.mean())
print(drop_df.Age.median())

# replace missing ages with median age since mean & median very close
# age most likely follows normal distribution
drop_df['Age'].fillna(drop_df.Age.median(), inplace=True)

# now lets drop unnecessary columns
drop_df = drop_df.drop(labels=['Cabin','Embarked'], axis=1)

# convert Sex column to binary
drop_df['Sex'] = drop_df['Sex'].map(lambda x: 1 if x == 'male' else 0)

drop_df.info()

final_df = drop_df.drop(labels=['Embark_S'], axis=1)

final_df.rename(index=str, columns={"Sex": "Male"}, inplace=True)

# A:
Y = final_df['Survived']
X = final_df.drop(labels='Survived', axis=1)

from sklearn.feature_selection import SelectKBest, chi2, f_classif

# A:

# To find features that are different(f-value) but related(chi squared score) to our target, then the features are meaningful

# f_classif is ANOVA F-value for categorical variables against continuous target
# testing for differences in variance
# if variance of variable and target are the same, then variable will be unable to predict target
# you cant use ticket fare to 
# h0 = variances equal, h1= variances different
# large value means variances are different which means variable is able reject null

# chi2 is Chi squared for categorical variables against categorical target
# similar to pearson correlation but for categorical variables
# testing for differences in observed and expected frequencies,
# if variable and target are independent, observe count close to expected count, which means no difference
# h0 = no difference, h1= have difference
# large value means variable non randomly related to target, reject null

skb_chi2 = SelectKBest(chi2, 5)
skb_f = SelectKBest(f_classif, 5)

# fit the data
skb_chi2.fit(X,Y)
skb_f.fit(X,Y)

kbest = pd.DataFrame([X.columns, list(skb_f.scores_), list(skb_chi2.scores_)], 
                     index=['features','f_classif','chi2 score']).T.sort_values('f_classif', ascending=False)
kbest

kbest_columns = kbest['features'].head(5).values
kbest_columns

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
selector = RFECV(lr, step=1, cv=10)
selector = selector.fit(X, Y)

print selector.support_
print selector.ranking_

rfecv_columns = np.array(X.columns)[selector.support_]
rfecv_columns

from sklearn.preprocessing import StandardScaler

# A:
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = pd.DataFrame(Xs, columns=X.columns)

from sklearn.linear_model import LogisticRegressionCV

# A:
lr_cv = LogisticRegressionCV(penalty='l1',Cs=100,solver='liblinear',cv=10)
lr_cv.fit(Xs,Y)

features = pd.DataFrame([X.columns, lr_cv.coef_[0],np.abs(lr_cv.coef_[0])], index=['features','coef','abs_coef'])
features = features.T.sort_values(by='abs_coef', ascending=False)
features.head(5)

lr_columns = features['features'].head(5).values
lr_columns

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# A:
print(kbest_columns)
print(rfecv_columns)
print(lr_columns)
print(X.columns)

optimal_lr = LogisticRegression(C=lr_cv.C_[0], penalty='l1')

# defining a function to test our best features head-to-head
def score(X):
    scores = cross_val_score(optimal_lr, X, Y, cv=5)
    return scores.mean(), scores.std()

# list of all our lists of best features being executed in the score function
all_scores = [
    score(X[kbest_columns]),
    score(X[rfecv_columns]),
    score(X[lr_columns]),
    score(Xs)]

#putting results into a dataframe
pd.DataFrame(all_scores, columns=['mean score', 'std score'], index = ['kbest', 'rfecv', 'lr', 'all'])

# Almost no difference for features selected from different feature selection methods

# A:
features.set_index('features', inplace=True)

features['coef'].sort_values().plot(kind='bar')



