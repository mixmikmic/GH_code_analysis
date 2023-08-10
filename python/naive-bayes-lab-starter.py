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
columns = np.array(cvt.get_feature_names())
# Complete the code
X_all

count_vect_df = pd.DataFrame(X_all.todense(), columns=cvt.get_feature_names())

df1 = count_vect_df.sum(axis=0)

plt.figure(figsize=(20,10))
df1[df1>50].sort_values(ascending=False).plot(kind='bar')

# Let's use the CountVectorizer to count words for us
cvt      =  CountVectorizer(strip_accents='unicode')
X_all    =  cvt.fit_transform(insults_df['Comment'])

# Complete the code

# look up the appropriate parameters
# CountVectorizer?

from nltk.corpus import stopwords
stop = stopwords.words('english')



# Here's the code -- you can adapt it from here on out.
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('cls', MultinomialNB())
]) 

pipeline.fit(tweets_train["TEXT"], tweets_train["LANG"])

# don't forget to score







# update the code to display the classification report
get_ipython().set_next_input('print classification_report');get_ipython().magic('pinfo classification_report')

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

# Using your pipeline, predict the probabilities of each language
# Then, call plot_roc

## Your code here to predict the probabilities of each class

# EXAMPLE of testing a particular language
# plot_roc(tweets_test['LANG'].apply(lambda x: x == "en"), predicted_proba[:, list(pipeline.classes_).index("en")])







