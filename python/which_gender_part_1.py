from bs4 import BeautifulSoup
from functools32 import lru_cache
from gensim import parsing, utils
from nltk.stem import WordNetLemmatizer

import numpy as np
import os
import pandas as pd
import re
import sys
import tarfile
import time
import traceback
import xml.etree.cElementTree as ET

dataset_filepath = "../../datasets/blogs_dataset_ages_24_25.tar.gz"
#dataset_filepath = "../../datasets/blogs_dataset_tiny.tar.gz"
dataset_filepath = "blogs_dataset_ages_24_25.tar.gz"
t0 = time.time()

def print_elapsed_time():
    print "\nElapsed Time :", "%.1f" % ((time.time() - t0)/60), "minutes to reach this point (from the start)"

# Read file contents

wnl = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=150000)(wnl.lemmatize)

stoplist = set("urllink".split())
stoplist.update(parsing.preprocessing.STOPWORDS)

def read_blog_file(blogfile):
    contents = " ".join(blogfile.readlines())
    blog = BeautifulSoup(contents, "lxml")
    # datelist = blog.findAll("date")
    # print(datelist)
    postlist = blog.findAll("post")
    # print(len(postlist))
    # TODO complete the implementation

def read_first_blog_post(blogfile):
    # TODO replace this with reading all posts
    first_post_text = None
    for event, elem in ET.iterparse(blogfile):
        # print("%5s, %4s, %10s" % (event, element.tag, element.text))
        if elem.tag == "post" and first_post_text is None:
            first_post_text = elem.text.strip()
            # Remove tabs
            first_post_text = re.sub("[\s+]", " ", first_post_text)
            break
    
    return first_post_text, preprocess_text(first_post_text.lower())

def preprocess_text(text):
    # Remove none English characters
    text = re.sub("[^a-zA-Z]", " ", text)
    words = [lemmatize(token) for token in utils.simple_preprocess(text) if
             token not in stoplist and len(token) > 1]
    return " ".join(words)

columns = ["blogger_id", "gender", "age", "industry", "astro_sign", "filename", "first_blog_post"]
df = pd.DataFrame(columns=columns)
tar=tarfile.open(dataset_filepath)
ctr = [0] * 2
for tarinfo in tar.getmembers():
    if os.path.splitext(tarinfo.name)[1] == ".xml":
        info = os.path.splitext(tarinfo.name)[0].split("/")[-1]
        tmp_df = pd.DataFrame(dict(zip(columns, info.split("."))), index=[0])
        tmp_df["filename"] = info + ".xml"
        blogfile = tar.extractfile(tarinfo)
        ctr[0] += 1
        #read_blog_file(blogfile)
        try : 
            text, preprocessed_text = read_first_blog_post(blogfile)
            blogfile.close()
        except Exception, e:
            text, preprocessed_text = None, None
            # traceback.print_exc(file=sys.stdout)
            ctr[1] += 1
            print info, "has problem,", str(e)
        tmp_df["first_blog_post"] = text
        tmp_df["first_blog_post_preprocessed"] = preprocessed_text
        df = pd.concat([df, tmp_df])

tar.close()
print ctr[0], "read"
print ctr[1], "has errors reading"

df = df.reset_index(drop=True)
print df.shape
print sys.getsizeof(df)
df.head()
df.shape
print_elapsed_time()

df.head()

print "Before pruning", pd.value_counts(df.gender)
df = df.dropna(subset=["first_blog_post"])
print "After pruning", pd.value_counts(df.gender)
df.head()

# Make a backup of the Pandas Dataframe
import datetime
fmt_str = "%Y%m%d_%H%M%S"
filename = "blog_dataset_df_"+datetime.datetime.now().strftime(fmt_str)
df.to_csv(filename,sep="\t", index=False, encoding="utf8")

# Drop columns not required for the BoW model
df.drop(["age", "blogger_id", "industry", "astro_sign", "filename", "first_blog_post"], axis=1, inplace=True)
# Replace gender with a numeric value
df["gender"] = df["gender"].replace({"female":2, "male":1})
# Tokenize the text, prepare datset
def my_func(x):
    text = x["first_blog_post_preprocessed"]
    return text.split() if text else []
df["tokenized_text"] = df.apply(my_func , axis=1)
X, y = np.array(df.tokenized_text.values.tolist()), np.array(df.gender.values.tolist())
df.head()

print df.shape
print len(X)
print len(y)

# Experiment with the vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer

min_dfs = [0.01, 0.005, 0.001]
tfidf_vec = [None] * len(min_dfs)
features = [None] * len(min_dfs)
ctr = 0
def get_topn_tfidf_vec_terms(vec, features, n=5):
    # Get the top n terms with highest tf-idf score
    # Credit : http://stackoverflow.com/a/34236002
    feature_array = np.array(vec.get_feature_names())
    tfidf_sorting = np.argsort(features.toarray()).flatten()[::-1]
    return feature_array[tfidf_sorting][:n]

for min_df in min_dfs:
    tfidf_vec[ctr] = TfidfVectorizer(analyzer=lambda x: x, min_df=min_df)
    features[ctr] = tfidf_vec[ctr].fit_transform(X)
    print features[ctr].shape[1],           "features for minimum document frequency %.1f%%\n" % (min_df * 100),           "top 8 terms", get_topn_tfidf_vec_terms(tfidf_vec[ctr], features[ctr], n=8), "\n"
    ctr += 1

# Next, run grid search to pick the best hyperparameters 
from operator import itemgetter
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import LinearSVC, SVC

def find_best_hyperparameters(clf, vectorizer, param_dist, num_iters=20):
    # Run the grid search
    print "Finding best hyperparameters for", clf.__class__.__name__
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=num_iters, n_jobs=7)
    random_search.fit(vectorizer.fit_transform(X), y)
    # Iterate through the scores and print the best 3
    top_scores = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[:3]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("\tMean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("\tParameters: {0}".format(score.parameters))
    print top_scores[0]
    return random_search.best_estimator_


best_rf = find_best_hyperparameters(RandomForestClassifier(random_state = 120), tfidf_vec[1],
                                    { "bootstrap": [True, False],
                                      "criterion": ["gini", "entropy"],
                                      "max_depth": np.arange(3, 11).tolist() + [None],
                                      "n_estimators": np.arange(50, 550, 50).tolist(),
                                      "random_state": np.arange(120, 12000, 240).tolist()
                                    },
                                    num_iters=20)
print best_rf,"\n"

best_et = find_best_hyperparameters(ExtraTreesClassifier(random_state = 9000), tfidf_vec[1],
                                    { "max_depth": np.arange(3, 11).tolist() + [None],
                                      "n_estimators": np.arange(50, 550, 50).tolist(),
                                      "random_state": np.arange(120, 12000, 240).tolist()
                                    },
                                    num_iters=20)
print best_et,"\n"

best_svm = find_best_hyperparameters(SVC(kernel="linear", random_state = 840), tfidf_vec[1],
                                    { "C" : np.arange(0.1, 1, 0.1).tolist(),
                                      "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, "auto"],
                                      "random_state": np.arange(120, 12000, 240).tolist(),
                                      "tol": [0.0001, 0.001, 0.01]
                                    },
                                    num_iters=20)
print best_svm,"\n"

best_lianersvc = find_best_hyperparameters(LinearSVC(random_state = 11640), tfidf_vec[1],
                                           { "C" : np.arange(0.1, 1, 0.1).tolist(),
                                             "loss": ["hinge", "squared_hinge"],
                                             "random_state": np.arange(120, 12000, 240).tolist(),
                                             "tol": [0.0001, 0.001, 0.01]
                                           },
                                           num_iters=20)
print best_lianersvc,"\n"

best_svm_rbf = find_best_hyperparameters(SVC(kernel="rbf", random_state = 600), tfidf_vec[1],
                                         { "C" : np.arange(0.1, 1, 0.1).tolist(),
                                           "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, "auto"],
                                           "random_state": np.arange(120, 12000, 240).tolist(),
                                           "tol": [0.0001, 0.001, 0.01]
                                         },
                                         num_iters=20)
print best_svm_rbf
print_elapsed_time()

# Next, train the models
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tabulate import tabulate

models = []
models.append(Pipeline([("tfidf_vec", tfidf_vec[0]), ("svm_klinear", SVC(kernel="linear"))]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("svm_klinear", SVC(kernel="linear"))]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[2]), ("svm_klinear", SVC(kernel="linear"))]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("best_et", best_et)]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("best_rf", best_rf)]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("best_svm_klinear", best_svm)]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("best_lianersvc", best_lianersvc)]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("best_svm_rbf", best_svm_rbf)]))
models.append(Pipeline([("tfidf_vec", tfidf_vec[1]), ("mnb", MultinomialNB())]))

models_with_desc = [
    ("SVM (Linear), TF-IDF, min_df 1.0%", models[0]),
    ("SVM (Linear), TF-IDF, min_df 0.5%", models[1]),
    ("SVM (Linear), TF-IDF, min_df 0.1%", models[2]),
    ("Extra Trees - 'Best', TF-IDF, min_df 0.5%", models[3]),
    ("Random Forest - 'Best', TF-IDF, min_df 0.5%", models[4]),
    ("SVM (Linear) - 'Best', TF-IDF, min_df 0.5%", models[5]),
    ("LinearSVC - 'Best', TF-IDF, min_df 0.5%", models[6]),
    ("SVM (RBF) - 'Best', TF-IDF, min_df 0.5%", models[7]),
    ("MultinomialNB, TF-IDF, min_df 0.5%", models[8])
]

def get_cv_scores(models_with_desc):
    cv_scores = []
    for model_id, model in models_with_desc:
        print "Training:", model_id
        ts = time.time()
        # cv_score = cross_val_score(model, X, y, cv=5, n_jobs=7).mean() # gives a pickling error
        cv_score = cross_val_score(model, X, y, cv=5).mean()
        cv_scores.append((model_id, cv_score))
        print "\tTime taken:", "%.1f" % ((time.time() - ts)/60), "minutes\n"
        print_elapsed_time()
    return cv_scores

scores = sorted([(model_id, cross_val_score(model, X, y, cv=5).mean()) 
                 for model_id, model in models_with_desc], 
                key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
print_elapsed_time()

# Read the GloVe word vector representation file
glove_small_50_filepath = "glove.6B.50d.txt"
glove_small_100_filepath = "glove.6B.100d.txt"
glove_small_200_filepath = "glove.6B.200d.txt"
glove_small_300_filepath = "glove.6B.300d.txt"

def read_GloVe_file(filepath):
    print "Reading", filepath
    glove_w2v = {}
    with open(filepath, "rb") as lines:
        for line in lines:
            parts = line.split()
            glove_w2v[parts[0]] = np.array(map(float, parts[1:]))
    print len(glove_w2v.keys()), "keys. First 5 :", glove_w2v.keys()[:5], "\n"
    return glove_w2v

glove_small_50_w2v = read_GloVe_file(glove_small_50_filepath)
glove_small_100_w2v = read_GloVe_file(glove_small_100_filepath)
glove_small_300_w2v = read_GloVe_file(glove_small_300_filepath)
print_elapsed_time()

X[:2]

# Test whether some of the words are present in the GloVe word vector representation file
for word in ["friday", "night", "music"]:
    if word in glove_small_50_w2v:
        print word, "\t", glove_small_50_w2v[word]

# Word vector equivalent of CountVectorizer & TfidfVectorizer (respectively)
# Each word in each blog post is mapped to its vector; 
# then this helper class computes the mean of those vectors
# Credit : https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb
from collections import defaultdict

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

tfidf_vec_glove_small_50 = TfidfEmbeddingVectorizer(glove_small_50_w2v)
tfidf_vec_glove_small_100 = TfidfEmbeddingVectorizer(glove_small_100_w2v)
tfidf_vec_glove_small_300 = TfidfEmbeddingVectorizer(glove_small_300_w2v)

models = []
models.append(Pipeline([("tfidf_vec_glove_small_50", tfidf_vec_glove_small_50), ("best_rf", best_rf)]))
models.append(Pipeline([("tfidf_vec_glove_small_50", tfidf_vec_glove_small_50), ("best_svm_klinear", best_svm)]))
models.append(Pipeline([("tfidf_vec_glove_small_50", tfidf_vec_glove_small_50), ("best_lianersvc", best_lianersvc)]))
models.append(Pipeline([("tfidf_vec_glove_small_50", tfidf_vec_glove_small_50), ("best_svm_rbf", best_svm_rbf)]))
models.append(Pipeline([("tfidf_vec_glove_small_100", tfidf_vec_glove_small_100), ("best_rf", best_rf)]))
models.append(Pipeline([("tfidf_vec_glove_small_100", tfidf_vec_glove_small_100), ("best_svm_klinear", best_svm)]))
models.append(Pipeline([("tfidf_vec_glove_small_100", tfidf_vec_glove_small_100), ("best_lianersvc", best_lianersvc)]))
models.append(Pipeline([("tfidf_vec_glove_small_100", tfidf_vec_glove_small_100), ("best_svm_rbf", best_svm_rbf)]))
models.append(Pipeline([("tfidf_vec_glove_small_300", tfidf_vec_glove_small_300), ("best_rf", best_rf)]))
models.append(Pipeline([("tfidf_vec_glove_small_300", tfidf_vec_glove_small_300), ("best_svm_klinear", best_svm)]))
models.append(Pipeline([("tfidf_vec_glove_small_300", tfidf_vec_glove_small_300), ("best_lianersvc", best_lianersvc)]))
models.append(Pipeline([("tfidf_vec_glove_small_300", tfidf_vec_glove_small_300), ("best_svm_rbf", best_svm_rbf)]))
#models.append(Pipeline([("tfidf_vec_glove_small_50", tfidf_vec_glove_small_50), ("mnb", ())])) # 
# NOTE : MultinomialNB will not work because of non-negative feature values

models_with_desc = [
    ("Random Forest - 'Best', TF-IDF GloVe small 50-Dim", models[0]),
    ("SVM (Linear) - 'Best',  TF-IDF GloVe small 50-Dim", models[1]),
    ("LinearSVC - 'Best',     TF-IDF GloVe small 50-Dim", models[2]),
    ("SVM (RBF) - 'Best',     TF-IDF GloVe small 50-Dim", models[3]),
    ("Random Forest - 'Best', TF-IDF GloVe small 100-Dim", models[4]),
    ("SVM (Linear) - 'Best',  TF-IDF GloVe small 100-Dim", models[5]),
    ("LinearSVC - 'Best',     TF-IDF GloVe small 100-Dim", models[6]),
    ("SVM (RBF) - 'Best',     TF-IDF GloVe small 100-Dim", models[7]),
    ("Random Forest - 'Best', TF-IDF GloVe small 300-Dim", models[8]),
    ("SVM (Linear) - 'Best',  TF-IDF GloVe small 300-Dim", models[9]),
    ("LinearSVC - 'Best',     TF-IDF GloVe small 300-Dim", models[10]),
    ("SVM (RBF) - 'Best',     TF-IDF GloVe small 300-Dim", models[11])
]

# scores = [] # Because we want to compare with the previous BoW model
scores.extend(get_cv_scores(models_with_desc))

scores = sorted(scores, key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
print_elapsed_time()

# Try 200-Dim word vectors from the GloVe small dataset
glove_small_200_filepath = "glove.6B.200d.txt"
glove_small_200_w2v = read_GloVe_file(glove_small_200_filepath)
tfidf_vec_glove_small_200 = TfidfEmbeddingVectorizer(glove_small_200_w2v)

models = []
models.append(Pipeline([("tfidf_vec_glove_small_200", tfidf_vec_glove_small_200), ("best_rf", best_rf)]))
models.append(Pipeline([("tfidf_vec_glove_small_200", tfidf_vec_glove_small_200), ("best_svm_klinear", best_svm)]))
models.append(Pipeline([("tfidf_vec_glove_small_200", tfidf_vec_glove_small_200), ("best_lianersvc", best_lianersvc)]))
models.append(Pipeline([("tfidf_vec_glove_small_200", tfidf_vec_glove_small_200), ("best_svm_rbf", best_svm_rbf)]))
models_with_desc = [
    ("Random Forest - 'Best', TF-IDF GloVe small 200-Dim", models[0]),
    ("SVM (Linear) - 'Best',  TF-IDF GloVe small 200-Dim", models[1]),
    ("LinearSVC - 'Best',     TF-IDF GloVe small 200-Dim", models[2]),
    ("SVM (RBF) - 'Best',     TF-IDF GloVe small 200-Dim", models[3]),
]
# scores = [] # Because we want to compare with the previous BoW model
scores.extend(get_cv_scores(models_with_desc))
scores = sorted(scores, key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))
print_elapsed_time()



