import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import nltk

profiles = pd.read_csv('./ready_for_cvec.csv', dtype=object, index_col= 0)

profiles.shape

profiles = profiles.dropna()

profiles.shape

profiles.sentiment_dummies.value_counts()

#tested all of the profile 'clean' levels, i.e. with different components removed/included
X = profiles.without_hashes
y = profiles.sentiment_dummies

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True,strip_handles=True)

classifier = LogisticRegression()

cvec = CountVectorizer(tokenizer= tokenizer.tokenize)

X = cvec.fit_transform(X)

cross_val_score(classifier, X, y)

#underpredicting negative sentiment from profiles
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')

#basic model testing function
def model_tester(X, y, model, return_vals=False):
    """Prints basic classification matrix and accuracy for a given model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        stratify=profiles['sentiment_dummies'], test_size=0.3)
    modeled = model()
    modeled.fit(X_train, y_train)
    predictions = modeled.predict(X_test)
    print(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
    print(accuracy_score(y_test, predictions))
    if return_vals:
        return X_train, X_test, y_train, y_test

#Baseline- if I predict 1 everytime, I have 75% accuracy
print(profiles['sentiment_dummies'].value_counts(), '\n')
print('BASELINE')
print(17548/profiles.shape[0])

model_tester(X, y, LogisticRegression)

model_tester(X, y, BernoulliNB)

model_tester(X, y, RandomForestClassifier)

bern = BernoulliNB()

param_grid = {'alpha': [ 1.1, 1.2, 1.3, 1.9, 2.2, 2.4], 'fit_prior':[True, False], 
             }
clf = GridSearchCV(bern, param_grid=param_grid)
clf.fit(X, y)

clf.best_score_

clf.best_estimator_

print(len(clf.predict(X_test)))
print(np.sum([int(x) for x in list(clf.predict(X_test))]))



tfidf = TfidfTransformer().fit(df)

df_tfidf = tfidf.transform(df)
ss = StandardScaler()
df_tfidf = ss.fit_transform(df_tfidf.toarray())

#marginally better than just the countvectorizer 
model_tester(df_tfidf, y, MultinomialNB)

#if building this model just on profiles, without pulling a sample of the user's tweets, drop
#weekday, hour
df = profiles[['number_of_people_they_follow',
          'number_of_user_tweets', 'user_followers_count', 'user_is_verified',
          'weekday', 'hour', 'profile_length', 'percent_in_dictionary', 'number_of_hashes']]

#converting datatypes

for col in list(df.columns):
    try: 
        df[col] = df[col].astype(int)
    except:
        df[col] = df[col].astype(float)

#standardizing 
df_ss = pd.DataFrame(ss.fit_transform(df), columns=list(df.columns))

Z_merged = pd.concat([Z, df_ss], axis=1)

model_tester(Z_merged, y, MultinomialNB)

pca = PCA()
pca.fit(df)

np.cumsum(pca.explained_variance_ratio_)

df_transformed = pca.transform(df)

col_headers = ['PC' + str(i) for i in range(1, 10)]
df_transformed = pd.DataFrame(df_transformed, columns=col_headers)
df_transformed = df_transformed.drop(['PC' + str(i) for i in range(3, 10)], axis=1)

Z_merged = pd.concat([Z, df_transformed], axis=1)

model_tester(df, y, MultinomialNB)

#LOOKING AT LSA (truncatedSVD)

# #Looked at reducing dimensionality, classifying positive more than 98% of time
# lsa = TruncatedSVD(n_components=100)
# lsa.fit(X)
# X = lsa.transform(X)

# #turning PCA into dataframe
# cols = ['PCA' + str(i) for i in range(1, 101)]
# df = pd.DataFrame(X, columns=cols)

# #very high scores, but... see below
# cross_val_score(classifier, df, y)

# #Seeing what was happening- predicted almost entirely positive sentiments
# lr = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(df, y)
# lr.fit(X_train, y_train)
# predictions = lr.predict(X_test)
# print(pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')

