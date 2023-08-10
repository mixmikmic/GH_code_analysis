import os
import pickle
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#set data path
LOCAL_DATA_PATH = 'C:\Users\JoAnna\political_history\processed_data'
SAVE_PATH = 'C:\Users\JoAnna\political_history\shibboleth\pkl_objects'
os.chdir(LOCAL_DATA_PATH)

#import data
labels = pickle.load(open('bow_labels.pkl', "r"))
text = pickle.load(open('bow_processed_text.pkl', "r"))

#train/test split of data (randomized)
text_train, text_test, labels_train, labels_test = cross_validation.train_test_split(text, labels, test_size=0.2, random_state=42)

print text_train[100]

#tfidf vectorizer and numpy array
vectorizer = TfidfVectorizer(sublinear_tf=True)
text_train_transformed = vectorizer.fit_transform(text_train).toarray()
text_test_transformed  = vectorizer.transform(text_test).toarray()

#test vectorizer
#print len(vectorizer.get_feature_names())
#feature_names = vectorizer.get_feature_names()
#feature_names[5000:5020]

#build classifier pipeline
select = SelectPercentile(f_classif)
pca = PCA()
feature_selection = FeatureUnion([('select', select), ('pca', pca)],
                    transformer_weights={'pca': 10})
clfNB = GaussianNB()

steps1 = [('feature_selection', feature_selection),
        ('naive_bayes', clfNB)]

pipeline1 = sklearn.pipeline.Pipeline(steps1)

#search for best parameters
parameters1 = dict(feature_selection__select__percentile=[.05, .1, .25], 
              feature_selection__pca__n_components=[10, 50, 100])

cv = sklearn.grid_search.GridSearchCV(pipeline1, param_grid=parameters1)

#because tf-idf vectorizer isn't in this pipeline, fit/predict on transformed data
cv.fit(text_train_transformed, labels_train)
pred = cv.predict(text_test_transformed)

print cv.best_params_

#pipeline.fit(features_train, labels_train)
#pred = pipeline.predict(features_test)
report = sklearn.metrics.classification_report(labels_test, pred)
print report

accuracy = sklearn.metrics.accuracy_score(labels_test, pred)
print accuracy

#set up scoring function and table
scoring_table = PrettyTable(['pipeline_name', 'accuracy', 'precision', 'recall', 'auc'])

def scoring_function(pipeline_name, test_labels, prediction):
    """
    runs evaluation metrics on prediction from classifier
    Args:
        labels from the test data set, prediction from classifier     
    Returns:
        prints scoring functions, appends scores to scoring dataframe
    """
    accuracy = sklearn.metrics.accuracy_score(test_labels, prediction)
    precision = sklearn.metrics.precision_score(test_labels, prediction)
    recall = sklearn.metrics.recall_score(test_labels, prediction)
    auc = sklearn.metrics.roc_auc_score(test_labels, prediction)
    print "Validation Metrics for %s: accuracy: %s, precision: %s, recall: %s, auc: %s"%(pipeline_name, accuracy, precision, recall, auc)
    
    scoring_table.add_row([pipeline_name, accuracy, precision, recall, auc])
    return scoring_table

#test scoring function using test classifier above
scoring_function('test1', labels_test, pred)
print scoring_table

# set-up generic grid-search cv function
def gridsearch_pipeline(pipeline_name, train_data, train_labels, test_data, pipeline_steps, parameters):
    """
    generic function to run gridsearchcv on an input dataset, pipeline, and parameters
    Args:
        data separated into features/labels and train/test
        steps of the pipeline
        parameters for gridsearchcv
    Returns:
        best parameters from gridsearch, prediction for test features
    """
    #pipeline
    pipe = sklearn.pipeline.Pipeline(pipeline_steps)
    
    #gridsearch
    cv = sklearn.grid_search.GridSearchCV(pipe, param_grid=parameters)
    cv.fit(train_data, train_labels)
    pred = cv.predict(test_data)
    print cv.best_params_
    return pred

#Put together pieces of classifier

#tf-idf vectorizer
vectorizer1 = TfidfVectorizer(sublinear_tf=True)
vectorizer2 = TfidfVectorizer(max_df = 1, min_df = 0, sublinear_tf=True)
vectorizer3 = TfidfVectorizer(ngram_range = (1,3), sublinear_tf=True)
vectorizer4 = TfidfVectorizer(max_df = 0.8, min_df = 0.2, ngram_range = (1,3), sublinear_tf=True)

#feature selection
select = SelectPercentile(f_classif)
pca = PCA()
feature_selection = FeatureUnion([('select', select), ('pca', pca)],
                    transformer_weights={'pca': 10})

#classifier
clfNB = GaussianNB()
clfAdaBoost = AdaBoostClassifier(random_state = 42)
clfLR = LogisticRegression(random_state=42, solver='sag')
clfSVM = SGDClassifier(loss='modified_huber', penalty='l2', n_iter=200, random_state=42)

#test2 - GaussianNB, simple vectorizer, PCA
steps = [
         ('feature_pick', pca),
         ('classifier', clfNB)]

params = dict(feature_pick__n_components=[100, 200, 500])

prediction = gridsearch_pipeline('test2', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test2', labels_test, prediction)
print scoring_table

#test3 - GaussianNB, simple vectorizer, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfNB)]

params = dict(feature_pick__percentile=[7, 10, 15])

prediction = gridsearch_pipeline('test3', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test3', labels_test, prediction)

#test4 - GaussianNB, simple vectorizer, Feature Union
steps = [
         ('feature_selection', feature_selection),
         ('classifier', clfNB)]

params = dict(feature_selection__select__percentile=[5, 10, 15], 
              feature_selection__pca__n_components=[50, 100, 200])

prediction = gridsearch_pipeline('test4', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test4', labels_test, prediction)

#test5 - AdaBoost, simple vectorizer, PCA
steps = [
         ('feature_pick', pca),
         ('classifier', clfAdaBoost)]

params = dict(feature_pick__n_components=[100, 200, 500],
              classifier__n_estimators=[10, 20, 50],
              classifier__learning_rate=[0.1, 1, 1.5])

prediction = gridsearch_pipeline('test5', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test5', labels_test, prediction)

#tes6 - AdaBoost, simple vectorizer, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfAdaBoost)]

params = dict(feature_pick__percentile=[5, 10, 20],
              classifier__n_estimators=[10, 20, 50],
              classifier__learning_rate=[0.1, 1, 1.5])

prediction = gridsearch_pipeline('test6', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test6', labels_test, prediction)

#test7 - adaboost, simple vectorizer, Feature Union
steps = [
         ('feature_selection', feature_selection),
         ('classifier', clfAdaBoost)]

params = dict(feature_selection__select__percentile=[5, 10, 15], 
              feature_selection__pca__n_components=[50, 100, 200],
              classifier__n_estimators=[10, 20, 50],
              classifier__learning_rate=[0.1, 1, 1.5])

prediction = gridsearch_pipeline('test7', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test7', labels_test, prediction)

#test8 - svm, simple vectorizer, PCA
steps = [
         ('feature_pick', pca),
         ('classifier', clfSVM)]

params = dict(feature_pick__n_components=[100, 200, 500],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test8', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test8', labels_test, prediction)

#test9 - svm, simple vectorizer, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfSVM)]

params = dict(feature_pick__percentile=[5, 10, 15],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test9', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test9', labels_test, prediction)

#test10 - svm, simple vectorizer, Feature Union
steps = [
         ('feature_selection', feature_selection),
         ('classifier', clfSVM)]

params = dict(feature_selection__select__percentile=[5, 10, 15], 
              feature_selection__pca__n_components=[100, 200, 500],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test10', text_train_transformed, labels_train, text_test_transformed, steps, params)
scoring_function('test10', labels_test, prediction)

#new vectorizer
text_train_transformed2 = vectorizer2.fit_transform(text_train).toarray()
text_test_transformed2  = vectorizer2.transform(text_test).toarray()
print len(text_train_transformed2)

#test11 - GaussianNB, vectorizer with frequency cutoffs, Feature Union
steps = [
         ('feature_pick', select),
         ('classifier', clfNB)]

params = dict(feature_pick__percentile=[7, 10, 15])

prediction = gridsearch_pipeline('test11', text_train_transformed2, labels_train, text_test_transformed2, steps, params)
scoring_function('test11', labels_test, prediction)

#test12 - svm, vectorizer with frequency cutoffs, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfSVM)]

params = dict(feature_pick__percentile=[5, 10, 15],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test12', text_train_transformed2, labels_train, text_test_transformed2, steps, params)
scoring_function('test12', labels_test, prediction)

#test13 - svm, vectorizer with frequency cutoffs, Feature Union
steps = [
         ('feature_selection', feature_selection),
         ('classifier', clfSVM)]

params = dict(feature_selection__select__percentile=[5, 10, 15], 
              feature_selection__pca__n_components=[100, 200, 500],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test13', text_train_transformed2, labels_train, text_test_transformed2, steps, params)
scoring_function('test13', labels_test, prediction)

#import data
text_nostop = pickle.load(open("bow_processed_text_nostop.pkl", "r"))

#train/test split of data (randomized)
text_train_nostop, text_test_nostop, labels_train_nostop, labels_test_nostop = cross_validation.train_test_split(text_nostop, labels, test_size=0.2, random_state=42)

#vectorizer with uni-, bi-, and tri-grams
text_train_transformed_nostop = vectorizer3.fit_transform(text_train_nostop).toarray()
text_test_transformed_nostop  = vectorizer3.transform(text_test_nostop).toarray()
print len(text_train_transformed_nostop)

#test14 - GaussianNB, vectorizer with ngrams, Feature Union
steps = [
         ('feature_pick', select),
         ('classifier', clfNB)]

params = dict(feature_pick__percentile=[7, 10, 15])

prediction = gridsearch_pipeline('test14', text_train_transformed_nostop, labels_train, text_test_transformed_nostop, steps, params)
scoring_function('test14', labels_test, prediction)

#test15 - svm, vectorizer with ngrams, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfSVM)]

params = dict(feature_pick__percentile=[12, 15, 17],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test15', text_train_transformed_nostop, labels_train_nostop, text_test_transformed_nostop, steps, params)
scoring_function('test15', labels_test_nostop, prediction)

#test16 - svm, vectorizer with ngrams, Feature Union
steps = [
         ('feature_selection', feature_selection),
         ('classifier', clfSVM)]

params = dict(feature_selection__select__percentile=[10, 15], 
              feature_selection__pca__n_components=[100, 200, 500],
              classifier__alpha=[0.0001, 0.001])

prediction = gridsearch_pipeline('test16', text_train_transformed_nostop, labels_train_nostop, text_test_transformed_nostop, steps, params)
scoring_function('test16', labels_test_nostop, prediction)

#try reducing dimensionality of matrix
text_train_transformed_nostop2 = vectorizer4.fit_transform(text_train_nostop).toarray()
text_test_transformed_nostop2  = vectorizer4.transform(text_test_nostop).toarray()

#test17 - svm, vectorizer with ngrams, selectPercentile
steps = [
         ('feature_pick', select),
         ('classifier', clfSVM)]

params = dict(feature_pick__percentile=[12, 15, 17],
              classifier__alpha=[0.0001, 0.00001, 0.001])

prediction = gridsearch_pipeline('test17', text_train_transformed_nostop2, labels_train, text_test_transformed_nostop2, steps, params)
scoring_function('test17', labels_test, prediction)

#export table
data = scoring_table.get_string()

with open('scoring_table.txt', 'wb') as f:
    f.write(data)

#read in data from in list form
text_lists = pickle.load(open("bow_processed_text_list.pkl", "r"))
#train/test split of data (randomized)
text_train_list, text_test_list, labels_train_list, labels_test_list = cross_validation.train_test_split(text_lists, labels, test_size=0.2, random_state=42)

#set up tokenizer for raw text

os.chdir(SAVE_PATH)

from sklearn.pipeline import Pipeline

#export test 15
model = Pipeline([ 
    ('vectorize', TfidfVectorizer(ngram_range = (1,3), sublinear_tf=True, lowercase=False)), 
    ('select', SelectPercentile(f_classif, percentile=15)), 
    ('classify', SGDClassifier(loss='modified_huber', penalty='l2', n_iter=200, random_state=42, alpha=0.0001)), 
])

# train the pipeline (note this calls fit_transform on all transformers and fit on the final estimator) 
model.fit(text_train_nostop, labels_train_nostop) 

# save the entire model 
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

prediction = model.predict(text_test_nostop)

report = sklearn.metrics.classification_report(labels_test, prediction)
print report

#load pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#test model on test data
prediction = model.predict(text_test_nostop)

report = sklearn.metrics.classification_report(labels_test, prediction)
print report



