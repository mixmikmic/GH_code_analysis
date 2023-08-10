from csv import DictReader
from time import time
from numpy import mean, std
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from nltk import stem
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

IN_FOLDER = 'categories/data/' # folder for train and test.csv files

# Read training and test data (from baseline code)
def read_data(name):
    text, targets = [], []

    with open(''.join([IN_FOLDER, '{}.csv']).format(name)) as f:
        for item in DictReader(f):
            text.append(item['text'])
            targets.append(item['category'])

    return text, targets

# Loading training and testing data
text_train, targets_train = read_data('train')
text_test, targets_test = read_data('test')

print('Training samples:', len(targets_train))
print('Testing samples:', len(targets_test))
print('Ttraining category count:', len(set(targets_train)))
print('Testing category count:', len(set(targets_test)),'\n')
print('Training set:', Counter(targets_train).most_common(),'\n')
print('Testing set:', Counter(targets_test).most_common())

# Compare wordclouds for a couple of categories
personal_text = " ".join([post for (post,label) in zip(text_train,targets_train) if label=='personal'])
relationships_text = " ".join([post for (post,label) in zip(text_train,targets_train) if label=='relationships'])

personal_cloud = WordCloud(stopwords=STOPWORDS).generate(personal_text)
relationships_cloud = WordCloud(stopwords=STOPWORDS).generate(relationships_text)


plt.figure(1)
plt.imshow(personal_cloud)
plt.title('Personal')
plt.axis("off")
plt.figure(2)
plt.imshow(relationships_cloud)
plt.title('Relationships')
plt.axis("off")

text_train, targets_train = read_data('train')
text_test, targets_test = read_data('test')

baseline_model = make_pipeline(TfidfVectorizer(), LogisticRegression(), ).fit(text_train, targets_train)

baseline_prediction = baseline_model.predict(text_test)
baseline_score_training = f1_score(targets_train, baseline_model.predict(text_train), average='macro')
baseline_score = f1_score(targets_test, baseline_prediction, average='macro')
previous_score = baseline_score
best_score = baseline_score

print('baseline macro f1:', baseline_score, 'baseline macro f1 (training):', baseline_score_training)
print(classification_report(targets_test, baseline_prediction))

class TextStats(BaseEstimator, TransformerMixin):
    """Returns the length of the post (as number of tokens) and the number of question marks"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [(len(tokenizer.tokenize(text)), text.count('?'))
                for text in posts]

# Latent Dirichlet Allocation
lda = Pipeline([('tf', CountVectorizer(strip_accents='unicode', stop_words='english')),
                ('lda',LatentDirichletAllocation(n_topics=13, max_iter=50, learning_method='online', #'online', 
                                                 learning_offset=50.,doc_topic_prior=.1, topic_word_prior=.01,
                                                 random_state=0))])
# All together
feats =  FeatureUnion([('text_len', TextStats()), ('lda', lda),
                       ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)))])

t0 = time()

lda = Pipeline([('tf', CountVectorizer(strip_accents='unicode', stop_words='english')),
                ('lda',LatentDirichletAllocation(n_topics=13, max_iter=50, learning_method='online', #'online', 
                                                 learning_offset=50.,doc_topic_prior=.1, topic_word_prior=.01,
                                                 random_state=0)), ])

feats =  FeatureUnion([('text_len', TextStats()), ('lda', lda), 
                       ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)))])

lr_model = make_pipeline(feats, LogisticRegression(C=1., penalty='l1', random_state=0), ).fit(text_train, targets_train)
lr_prediction = lr_model.predict(text_test)

print('training score:', f1_score(targets_train, lr_model.predict(text_train), average='macro'))
lr_macro_f1 = f1_score(targets_test, lr_prediction, average='macro')
print('testing score:', lr_macro_f1)

print("done in %0.3fs." % (time() - t0))
print(classification_report(targets_test, lr_prediction))

stemmer = stem.PorterStemmer()

def porter_stem(sentence):
    # Porter stemming 
    stemmed_sequence = [stemmer.stem(word) for word in tokenizer.tokenize(sentence)
                        if word not in stopwords.words('english')]
    return ' '.join(stemmed_sequence)

stemmed_train = [porter_stem(post) for post in text_train]
stemmed_test = [porter_stem(post) for post in text_test]

t0 = time()


lda = Pipeline([('tf', CountVectorizer(strip_accents='unicode', stop_words='english')),
                ('lda',LatentDirichletAllocation(n_topics=13, max_iter=20, learning_method='online', 
                                                 learning_offset=50.,doc_topic_prior=.1, topic_word_prior=.01,
                                                 random_state=0)), ])

lda_tfidf_features =  FeatureUnion([('lda', lda),
                                    ('tfidf', TfidfVectorizer(strip_accents='unicode', min_df=4))])

stem_lr_model = make_pipeline(lda_tfidf_features, LogisticRegression(C=1.5, penalty='l1', random_state=0)
                             ).fit(stemmed_train, targets_train)
stem_lr_prediction = stem_lr_model.predict(stemmed_test)

print('training score:', f1_score(targets_train, stem_lr_model.predict(stemmed_train), average='macro'))
stem_lr_macro_f1 = f1_score(targets_test, stem_lr_prediction, average='macro')
print('testing score:', stem_lr_macro_f1)

print("done in %0.3fs." % (time() - t0))
print(classification_report(targets_test, stem_lr_prediction))

t0 = time()

stem_xgb_model = make_pipeline(TfidfVectorizer(strip_accents='unicode', min_df=5, ngram_range=(1, 2)), 
                               xgb.XGBClassifier(max_depth=10, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, 
                                                 gamma=.01, reg_alpha=4, objective='multi:softmax')
                              ).fit(stemmed_train, targets_train) 

stem_xgb_prediction = stem_xgb_model.predict(stemmed_test)

stem_xgb_macro_f1 = f1_score(targets_test, stem_xgb_prediction, average='macro')

print('training score:', f1_score(targets_train, stem_xgb_model.predict(stemmed_train), average='macro'))
stem_xgb_macro_f1 = f1_score(targets_test, stem_xgb_prediction, average='macro')
print('testing score:', stem_xgb_macro_f1)

print("done in %0.3fs." % (time() - t0))
print(classification_report(targets_test, stem_xgb_prediction))

def majority_element(a):
    c = Counter(a)
    value, count = c.most_common()[0]
    if count > 1:
        return value
    else:
        return a[0]

merged_predictions = [[s[0],s[1],s[2]] for s in zip(stem_lr_prediction, lr_prediction, stem_xgb_prediction)]
majority_prediction = [majority_element(p) for p in merged_predictions]

print('majority vote ensemble:', f1_score(targets_test, majority_prediction, average='macro')) 
print(classification_report(targets_test, majority_prediction))

classifiers = ['Baseline','Logistic Regression with stemming', 'Logistic Regression', 'XGB with stemming', 'Majority voting ensemble']
predictions = (baseline_prediction, stem_lr_prediction, lr_prediction, stem_xgb_prediction, majority_prediction)
for pred, clfs in zip(predictions, classifiers):
    print(''.join((clfs,':')))
    print('macro:',f1_score(targets_test, pred, average='macro'))
    print('weighted:',f1_score(targets_test, pred, average='weighted'))
    print('micro:',f1_score(targets_test, pred, average='micro'))
    print()



