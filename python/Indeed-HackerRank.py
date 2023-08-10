# Import the pandas package, then use the "read_csv" function to read
# the labeled training data

import pandas as pd       
train = pd.read_csv("train.tsv", header=0,                     delimiter="\t", quoting=3)

test = pd.read_csv("test.tsv", header=0,                     delimiter="\t", quoting=3)

print "Data read successfully!"
print "Number of train points:", len(train)
print "Number of test points:", len(test)

# Print the column headers
print "\n"
print "Train file headers:"
print "\n"
print train.dtypes.index

print "\n"
print "Test file headers:"
print "\n"
print test.dtypes.index

# Examine a typical job description

example1 = train["description"][0]
print example1

# Remove punctuation and weird characters (e.g. )
import re

# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z0-9]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1)  # The text to search
print letters_only

# Convert all to lower case and split words

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words
print words

# Remove english stop words

from nltk.corpus import stopwords # Import the stop word list
print stopwords.words("english") 

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print words

# Define a function to clean up the text input, combining the previously evaluated methods

from bs4 import BeautifulSoup

def job_to_words( raw_job ):
    # Function to convert a raw job posting to a string of words
    # The input is a single string (a raw job description), and 
    # the output is a single string (a preprocessed job description)
    #
    # 1. Remove HTML
    job_text = BeautifulSoup(raw_job, "lxml").get_text() 
    #
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z0-9]", " ", job_text) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 3. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 5. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

# Check the output on an example

clean_job = job_to_words( example1 )
print clean_job

# Get the number of jobs based on the dataframe column size
num_jobs = train["description"].size

# Initialize an empty list to hold the clean jobs
clean_train_jobs = []

print "Cleaning and parsing the training set job descriptions...\n"

# Loop over each job; create an index i that goes from 0 to the length
# of the job list 
for i in xrange( 0, num_jobs ):
    # Call our function for each one, and add the result to the list of

    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Job %d of %d\n" % ( i+1, num_jobs )                                                                    

    # clean jobs
    clean_train_jobs.append( job_to_words( train["description"][i] ) )    

print clean_train_jobs[0]
print train["tags"][0]

# Look for instances where number is followed by 'year..':

number_list=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
i = 0
for word in clean_train_jobs[0]:
    if word in number_list:
        print "found it"
    else:
        i+=1
print i

print "Creating the bag of words for descriptions...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer1 = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 2000,
                              ngram_range = (1,4)) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer1.fit_transform(clean_train_jobs)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Use a similar approach to vectorize the training labels

# Get the number of jobs based on the dataframe column size
num_data = train["tags"].size

# Initialize an empty list to hold the clean job labels
train_labels = []

print "Cleaning and parsing the training set job labels...\n"

# Loop over each job; create an index i that goes from 0 to the length
# of the job list 
for i in xrange( 0, num_data ):
    # Call our function for each one, and add the result to the list of

    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Job %d of %d\n" % ( i+1, num_data )                                                                    

    # clean out NaN
    if type(train["tags"][i]) != str:
        train["tags"][i] = ''
    train_labels.append(train["tags"][i]) 

print "Sample training labels: \n", train_labels[0]

print "Creating the bag of words for job labels...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Keep words hyphenated
pattern = "(?u)\\b[\\w-]+\\b"

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer2 = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000,
                             token_pattern=pattern) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_labels = vectorizer2.fit_transform(train_labels)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_labels = train_data_labels.toarray()

print "Sample vectorized training labels: \n",train_data_labels[0]

print "Confirm training data description shape: \n", train_data_features.shape
print "Confirm training data labels shape: \n",train_data_labels.shape

# Take a look at the words in the job description vocabulary

vocab_description = vectorizer1.get_feature_names()
print "Job label vocabulary: \n", vocab_description

# Take a look at the words in the job label vocabulary (should be the 12 labels)

vocab = vectorizer2.get_feature_names()
print "Job label vocabulary: \n", vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_labels, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

# Evaluate a few classifiers, then choose one for optimizing next

# Import the desired classifiers, splitters, metrics etc.

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from time import time

# All classifiers are named clf for compatibility with tester.py
# Comment out ('#') all classifiers other than the desired one

#clf = DecisionTreeClassifier(random_state=42)
#clf = KNeighborsClassifier()
clf = RandomForestClassifier(random_state=42)

# Split data into training and testing sets, using 30% split

t0 = time()

features_train, features_test, labels_train, labels_test =     train_test_split(train_data_features, train_data_labels, test_size=0.3, random_state=42)
    
clf.fit(features_train,labels_train)
labels_train_est = clf.predict(features_train)
labels_pred = clf.predict(features_test)

print "Results for Training: \n", classification_report(labels_train, labels_train_est)
print "\n"
print "Results for Testing: \n", classification_report(labels_test, labels_pred)

print "total train/test/prediction time:", round(time()-t0, 3), "s"

# This code block was used to optimize the model
# Various pipe components were commented / uncommented to test their effects
# The process and interim results are discussed in the accompanying report

# Import the necessary libaries

from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

t0 = time()

# Build estimator from PCA and Univariate selection:

#combined_features = FeatureUnion([('pca', PCA()), ('select', SelectKBest())])

# Use combined features to transform dataset:

#X_features = combined_features.fit(features,labels).transform(features)

# Piping: combine scaling, feature selection, PCA and classification
# into a single pipeline

#algo = DecisionTreeClassifier(random_state=42)
algo = RandomForestClassifier(random_state=42)

pipe = Pipeline([
#        ('scaler',MinMaxScaler()),
#        ('select',SelectKBest()),
#                 ('reduce_dim', PCA()),
#        ('features',combined_features),
                 ('algo',algo)
    ])

# Cross Validation - choose parameters that maximize the F1 score

# Parameter grid

para = {
#    'select__k':[23],
#    'select__k':np.arange(22,24),
#    'reduce_dim__n_components':np.arange(1,15),
#    'features__pca__n_components':[1, 2, 3],
#    'features__select__k':np.arange(1,24),
#    'algo__criterion': ["gini"],
#    'algo__criterion': ["gini","entropy"],
#    'algo__min_samples_split': [10],
#    'algo__min_samples_split': [2, 10, 20],
#    'algo__min_samples_split': np.arange(2,6),
#    'algo__max_depth': [9],
#    'algo__max_depth': [None, 2, 5, 10],
#    'algo__max_depth': np.arange(8, 15),
#    'algo__n_estimators': np.arange(8,12)],
#    'algo__n_estimators': [9,15,25,50,75,100,125],
    'algo__n_estimators': [9],
#    'algo__max_features': ['auto',None,2,5,10],
    'algo__max_features': [None],
#    'algo__criterion': ["gini","entropy"],
#    'algo__max_depth': [None,2,5,10],
#    'algo__min_samples_split': np.arange(1,5),
#    'algo__min_samples_leaf': np.arange(1,5),
#    'algo__min_weight_fraction_leaf': [0,0.05,0.1,0.2,0.5],
#    'algo__max_leaf_nodes':[None,2,3,5],
    'algo__n_jobs': [-1],
#    'algo__oob_score': [True,False]
#    'algo__min_samples_leaf': [1],
#    'algo__min_samples_leaf': [1, 5, 10],
#    'algo__min_samples_leaf': np.arange(1,3),
#    'algo__class_weight':["balanced"],
#    'algo__class_weight':["balanced",None],
#    'algo__max_leaf_nodes':[None,2,5,10],
#    'algo__max_leaf_nodes':np.arange(7,11),
#    'algo__splitter': ["random"]
#    'algo__splitter': ["best","random"]
       }

# Because of the small size of the dataset, use stratified shuffle split cross validation
# I found that 50 splits provided scores that closely matched the tester.py results and also
# kept runtimes to relatively reasonable durations

sss = StratifiedShuffleSplit(train_data_labels, 10, random_state = 42)
#cv_clf = GridSearchCV(pipe,param_grid=para, cv = sss, scoring='f1_weighted')

# Use 20% holdout for post CV testing

features_train, features_test, labels_train, labels_test =     train_test_split(train_data_features, train_data_labels, test_size=0.05, random_state=42)

sss = StratifiedShuffleSplit(labels_train, 10, random_state = 42)
cv_clf = GridSearchCV(pipe,param_grid=para, cv = sss, scoring='f1_weighted')

# Run CV on the training subset only
cv_clf.fit(features_train,labels_train)
clf = cv_clf.best_estimator_

print "model build and validation time:", round(time()-t0, 3), "s"
print '\n'
print "Best F1 score: %0.3f" % cv_clf.best_score_
print '\n'
print "Best Parameters:"
print '\n'
print cv_clf.best_params_
print '\n'

# Split data into training and testing sets, using 30% split

t0 = time()

#features_train, features_test, labels_train, labels_test = \
#    train_test_split(train_data_features, train_data_labels, test_size=0.3, random_state=42)
    
#clf.fit(features_train,labels_train)
labels_train_est = clf.predict(features_train)
labels_full_est = clf.predict(train_data_features)
labels_pred = clf.predict(features_test)

print "Results on Training set: \n", classification_report(labels_train, labels_train_est)
print "\n"
print "Results on Full set: \n", classification_report(train_data_labels, labels_full_est)
print "\n"
print "Results on Testing set: \n", classification_report(labels_test, labels_pred)

print "total train/test/prediction time:", round(time()-t0, 3), "s"

# Read the test data

# Verify that there are 2,921 rows and 1 column
print test.shape

# Create an empty list and append the clean jobs one by one
num_test = len(test["description"])
clean_test = [] 

print "Cleaning and parsing the test set job descriptions...\n"
for i in xrange(0,num_test):
    if( (i+1) % 500 == 0 ):
        print "Job %d of %d\n" % (i+1, num_test)
    raw_test = job_to_words( test["description"][i] )
    clean_test.append( raw_test )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer1.transform(clean_test)
test_data_features = test_data_features.toarray()

# Use the classifier to make job label predictions
result = clf.predict(test_data_features)

# Compare average tags per job for train and test sets
# I would expect the average to be similar for each set

train_tagged = np.sum(train_data_labels)
test_tagged = np.sum(result)

train_size = 4375
test_size = 2921

print "tags per train set:", (float(train_tagged) / train_size)
print "tags per test set:", (float(test_tagged) / test_size)

print "Difference (tags per train - tags per test):",(float(train_tagged) 
                                                     / train_size) - (float(test_tagged) / test_size)

# Convert numerical labels back to actual descriptions

num_labels = len(vocab)
num_test = len(result)

test_tags = []
for i in range(0,num_test):
    row = []
    for j in range(0,num_labels):
        if result[i][j] >= 0.5:
            row.append(vocab[j])
        b = ' '.join(row)
    test_tags.append(b)

# Confirm the test tag contains all 2,921 test points

print len(test_tags)

# Copy the results to a pandas dataframe with a "tags" header

output = pd.DataFrame( data={"tags":test_tags})

# Use pandas to write the tab-separated output file

output.to_csv( "tags.tsv",sep='\t',index=False)



