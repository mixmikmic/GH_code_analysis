import pandas as pd, seaborn as sns, numpy as np, matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

get_ipython().magic('matplotlib inline')

sns.set_style("darkgrid")

tweets_df = pd.read_csv("datasets/tweets_language.csv", encoding="utf-8", index_col=0)
tweets_df.index = tweets_df.index.astype(int)    # By default, everything read in is a string!

tweets_df.info()

# Note above that some rows are null, which we cannot use for training
tweets_df = tweets_df.dropna()

tweets_df.head()

# Let's use the CountVectorizer to count words for us
cvt      =  CountVectorizer(strip_accents='unicode', ngram_range=(1,1))
X_all    =  cvt.fit_transform(tweets_df['TEXT'])
columns  =  np.array(cvt.get_feature_names())          # ndarray (for indexing below)

# note this is a large sparse matrix. 
#    - rows are tweets, columns are words 
X_all

# Converting X_all toarray() may use too much memory (particularly for 32-bit Python!)
print X_all.shape
print "Requires {} ints to do a .toarray()!".format(X_all.shape[0] * X_all.shape[1])

# x_df     =  pd.DataFrame(X_all.toarray(), columns=columns)
# tf_df    =  pd.DataFrame(x_df.sum(), columns=["freq"])
# tf_df.sort_values("freq", ascending=False).head(10)

# So .. we'll use np.sum() to convert it directly from the sparse matrix!
# This is enormously more memory-efficient ...
#   It only requires one int per column since summing across columns is the total word count.

def get_freq_words(sparse_counts, columns):
    # X_all is a sparse matrix, so sum() returns a 'matrix' datatype ...
    #   which we then convert into a 1-D ndarray for sorting
    word_counts = np.asarray(X_all.sum(axis=0)).reshape(-1)

    # argsort() returns smallest first, so we reverse the result
    largest_count_indices = word_counts.argsort()[::-1]

    # pretty-print the results! Remember to always ask whether they make sense ...
    freq_words = pd.Series(word_counts[largest_count_indices], 
                           index=columns[largest_count_indices])

    return freq_words


freq_words = get_freq_words(X_all, columns)
freq_words[:20]

from sklearn.preprocessing import StandardScaler, minmax_scale

def hist_counts(word_counts):
    hist_counts = pd.Series(minmax_scale(word_counts), 
                            index=word_counts.index)
    
    # Overall graph is hard to understand, so let's break it into three graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6))
    
    hist_counts.plot(kind="hist", bins=50, ax=axes[0], title="Histogram - All")
    
    # look at the range of extreme commons that seem to exist below .01
    hist_counts[hist_counts < .01].plot(kind="hist", ax=axes[2], title="Histogram - Counts < .01")
    
    # There are a lot of really common tokens within 10% -- filter them out
    hist_counts[hist_counts > .1].plot(kind="hist", bins=50, ax=axes[1], title="Histogram - Counts > .1")

print(freq_words[:10])
hist_counts(freq_words)

cvt      =  CountVectorizer(strip_accents='unicode', stop_words="english", ngram_range=(1,1))
X_all    =  cvt.fit_transform(tweets_df['TEXT'])
columns  =  np.array(cvt.get_feature_names())

freq_words = get_freq_words(X_all, columns)

print(freq_words[:10])
hist_counts(freq_words)

# Checking range between .99 - .99999 -- there seem to be lots of words there
freq_words.quantile(.99999)

# find the %1, and %10 threshold for masking
freq_words[(freq_words >= 10) & (freq_words <= 150)]

cvt = CountVectorizer(stop_words="english", ngram_range=(2,4))
X_all = cvt.fit_transform(tweets_df['TEXT'])
columns  =  np.array(cvt.get_feature_names())

freq_words = get_freq_words(X_all, columns)
freq_words

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop += ['http', 'https', 'rt']

# These look pretty clean for a first step in anlaysis
cvt = CountVectorizer(stop_words=stop, lowercase=True, strip_accents="unicode", ngram_range=(1,2))
X_all = cvt.fit_transform(tweets_df['TEXT'])
columns  =  np.array(cvt.get_feature_names())

freq_words = get_freq_words(X_all, columns)
freq_words[:20]

# Find our training size
training_size = int(tweets_df.shape[0] * .7)

# Randomly sample our training data
tweets_train = tweets_df.sample(n=training_size, replace=True)

# Capture the rest of the dataset that's not "training" using an inverse mask (rows NOT IN training dataframe)
mask = tweets_df.index.isin(tweets_train.index)
tweets_test = tweets_df[~mask]

# Should be (2762, 1963) = training / testing = 70/30
tweets_train.shape[0], tweets_test.shape[0]

# MultinomialNB
pipeline = Pipeline([
    ('vect', CountVectorizer(lowercase=True, strip_accents='unicode', stop_words=stop)),
    ('tfidf', TfidfTransformer()),
    ('cls', MultinomialNB())
]) 
pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])
predicted = pipeline.predict(tweets_test["TEXT"])
pipeline.score(tweets_test["TEXT"], tweets_test["LANG"])

# Alternative -- train on all data
# MultinomialNB
pipeline = Pipeline([
    ('vect', cvt),
    # ('tfidf', TfidfTransformer()),
    ('cls', MultinomialNB())
]) 
pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])
predicted = pipeline.predict(tweets_test["TEXT"])
pipeline.score(tweets_test["TEXT"], tweets_test["LANG"])

# BernoulliNB
pipeline = Pipeline([
    ('vect', cvt),
    ('tfidf', TfidfTransformer()),
    ('cls', BernoulliNB())
]) 
pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])
predicted = pipeline.predict(tweets_test["TEXT"])
pipeline.score(tweets_test["TEXT"], tweets_test["LANG"])

# LogisticRegression
pipeline = Pipeline([
    ('vect', cvt),
    ('tfidf', TfidfTransformer()),
    ('cls', LogisticRegression())
]) 
pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])
predicted = pipeline.predict(tweets_test["TEXT"])
pipeline.score(tweets_test["TEXT"], tweets_test["LANG"])

# BernoulliNB
cvt2 = CountVectorizer(stop_words=stop, lowercase=True, strip_accents="unicode", ngram_range=(1,2))
pipeline = Pipeline([
    ('vect', cvt2),
    # ('tfidf', TfidfTransformer()),
    ('cls', LogisticRegression())
]) 
pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])
predicted = pipeline.predict(tweets_test["TEXT"])
pipeline.score(tweets_test["TEXT"], tweets_test["LANG"])

tweets_test.shape, len(predicted)

predicted = pipeline.predict(tweets_test["TEXT"])
print classification_report(tweets_test["LANG"], predicted)

def multi_roc(y, probs):
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

from sklearn.metrics import roc_curve

def plot_roc(y, probs, threshmarkers=None):
    fpr, tpr, thresh = roc_curve(y, probs)

    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, lw=2)
   
    plt.xlabel("False Positive Rate\n(1 - Specificity)")
    plt.ylabel("True Positive Rate\n(Sensitivity)")
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xticks(np.linspace(0, 1, 21), rotation=45)
    plt.yticks(np.linspace(0, 1, 21))
    plt.show()

predicted_proba = pipeline.predict_proba(tweets_test['TEXT'])
plot_roc(tweets_test['LANG'].apply(lambda x: x == "en"), predicted_proba[:, list(pipeline.classes_).index("en")])
plot_roc(tweets_test['LANG'].apply(lambda x: x == "it"), predicted_proba[:, list(pipeline.classes_).index("it")])


(tweets_train.LANG.value_counts() / len(tweets_train)).mean()

predicted = pipeline.predict(tweets_df["TEXT"])

# Incorrectly classified
incorrect_preds = tweets_df[(predicted != tweets_df['LANG'])]

incorrect_df = pd.DataFrame({'actual': incorrect_preds['LANG'], 
                             'predicted': predicted[incorrect_preds.index],
                             'TEXT': incorrect_preds['TEXT']})

incorrect_df



