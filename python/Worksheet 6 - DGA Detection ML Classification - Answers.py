# Load Libraries - Make sure to run this cell!
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn import feature_extraction, tree, model_selection, metrics
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

df_final = pd.read_csv('../data/dga_features_final_df.csv')
print(df_final.isDGA.value_counts())
df_final.head()

# Load dictionary of common english words from part 1
from six.moves import cPickle as pickle
with open('../data/d_common_en_words' + '.pickle', 'rb') as f:
        d = pickle.load(f)

target = df_final['isDGA']
feature_matrix = df_final.drop(['isDGA'], axis=1)
print('Final features', feature_matrix.columns)

print( feature_matrix.head())

# Simple Cross-Validation: Split the data set into training and test data
feature_matrix_train, feature_matrix_test, target_train, target_test = model_selection.train_test_split(feature_matrix, target, test_size=0.25, random_state=33)

feature_matrix_train.count()

feature_matrix_test.count()

target_train.head()

# Train the decision tree based on the entropy criterion
clf = tree.DecisionTreeClassifier()  # clf means classifier
clf = clf.fit(feature_matrix_train, target_train)

# Extract a row from the test data
test_feature = feature_matrix_test[192:193]
test_target = target_test[192:193]

# Make the prediction
pred = clf.predict(test_feature)
print('Predicted class:', pred)
print('Accurate prediction?', pred[0] == test_target)

feature_matrix_test

# For simplicity let's just copy the needed function in here again

def H_entropy (x):
    # Calculate Shannon Entropy
    prob = [ float(x.count(c)) / len(x) for c in dict.fromkeys(list(x)) ] 
    H = - sum([ p * np.log2(p) for p in prob ]) 
    return H

def vowel_consonant_ratio (x):
    # Calculate vowel to consonant ratio
    x = x.lower()
    vowels_pattern = re.compile('([aeiou])')
    consonants_pattern = re.compile('([b-df-hj-np-tv-z])')
    vowels = re.findall(vowels_pattern, x)
    consonants = re.findall(consonants_pattern, x)
    try:
        ratio = len(vowels) / len(consonants)
    except: # catch zero devision exception 
        ratio = 0  
    return ratio

# ngrams: Implementation according to Schiavoni 2014: "Phoenix: DGA-based Botnet Tracking and Intelligence"
# http://s2lab.isg.rhul.ac.uk/papers/files/dimva2014.pdf

def ngrams(word, n):
    # Extract all ngrams and return a regular Python list
    # Input word: can be a simple string or a list of strings
    # Input n: Can be one integer or a list of integers 
    # if you want to extract multipe ngrams and have them all in one list
    
    l_ngrams = []
    if isinstance(word, list):
        for w in word:
            if isinstance(n, list):
                for curr_n in n:
                    ngrams = [w[i:i+curr_n] for i in range(0,len(w)-curr_n+1)]
                    l_ngrams.extend(ngrams)
            else:
                ngrams = [w[i:i+n] for i in range(0,len(w)-n+1)]
                l_ngrams.extend(ngrams)
    else:
        if isinstance(n, list):
            for curr_n in n:
                ngrams = [word[i:i+curr_n] for i in range(0,len(word)-curr_n+1)]
                l_ngrams.extend(ngrams)
        else:
            ngrams = [word[i:i+n] for i in range(0,len(word)-n+1)]
            l_ngrams.extend(ngrams)
#     print(l_ngrams)
    return l_ngrams

def ngram_feature(domain, d, n):
    # Input is your domain string or list of domain strings
    # a dictionary object d that contains the count for most common english words
    # finally you n either as int list or simple int defining the ngram length
    
    # Core magic: Looks up domain ngrams in english dictionary ngrams and sums up the 
    # respective english dictionary counts for the respective domain ngram
    # sum is normalized
    
    l_ngrams = ngrams(domain, n)
#     print(l_ngrams)
    count_sum=0
    for ngram in l_ngrams:
        if d[ngram]:
            count_sum+=d[ngram]
    try:
        feature = count_sum/(len(domain)-n+1)
    except:
        feature = 0
    return feature
    
def average_ngram_feature(l_ngram_feature):
    # input is a list of calls to ngram_feature(domain, d, n)
    # usually you would use various n values, like 1,2,3...
    return sum(l_ngram_feature)/len(l_ngram_feature)

def is_dga(domain, clf, d):
    # Function that takes new domain string, trained model 'clf' as input and
    # dictionary d of most common english words
    # returns prediction
    
    domain_features = np.empty([1,5])
    # order of features is ['length', 'digits', 'entropy', 'vowel-cons', 'ngrams']
    domain_features[0,0] = len(domain)
    pattern = re.compile('([0-9])')
    domain_features[0,1] = len(re.findall(pattern, domain))
    domain_features[0,2] = H_entropy(domain)
    domain_features[0,3] = vowel_consonant_ratio(domain)
    domain_features[0,4] = average_ngram_feature([ngram_feature(domain, d, 1), 
                                                  ngram_feature(domain, d, 2), 
                                                  ngram_feature(domain, d, 3)])
    pred = clf.predict(domain_features)
    return pred[0]


print('Predictions of domain %s is [0 means legit and 1 dga]: ' %('spardeingeld'), is_dga('spardeingeld', clf, d))  
print('Predictions of domain %s is [0 means legit and 1 dga]: ' %('google'), is_dga('google', clf, d)) 
print('Predictions of domain %s is [0 means legit and 1 dga]: ' %('1vxznov16031kjxneqjk1rtofi6'), is_dga('1vxznov16031kjxneqjk1rtofi6', clf, d)) 
print('Predictions of domain %s is [0 means legit and 1 dga]: ' %('lthmqglxwmrwex'), is_dga('lthmqglxwmrwex', clf, d)) 

# fair approach: make prediction on test data portion
target_pred = clf.predict(feature_matrix_test)
print(metrics.accuracy_score(target_test, target_pred))
print('Confusion Matrix\n', metrics.confusion_matrix(target_test, target_pred))

# Classification Report...neat summary
print(metrics.classification_report(target_test, target_pred, target_names=['legit', 'dga']))

# short-cut
clf.score(feature_matrix_test, target_test)

cvKFold = model_selection.KFold(n_splits=3, shuffle=True, random_state=33) 
cvKFold.get_n_splits(feature_matrix)

scores = model_selection.cross_val_score(clf, feature_matrix, target, cv=cvKFold)
print(scores)

# Get avergage score +- Standard Error (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html)
from scipy.stats import sem
def mean_score( scores ):
    return "Mean score: {0:.3f} (+/- {1:.3f})".format( np.mean(scores), sem( scores ))
print( mean_score( scores))

# These libraries are used to visualize the decision tree and require that you have GraphViz
# and pydot or pydotplus installed on your computer.

from sklearn.externals.six import StringIO  
from IPython.core.display import Image
import pydotplus as pydot


dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data, feature_names=['length', 'digits', 'entropy', 'vowel-cons', 'ngrams']) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#Create the Random Forest Classifier
random_forest_clf = RandomForestClassifier(n_estimators=10, 
                             max_depth=None, 
                             min_samples_split=2, 
                             random_state=0)

random_forest_clf = random_forest_clf.fit(feature_matrix_train, target_train)

#Next, create the SVM classifier
svm_classifier = svm.SVC()
svm_classifier = svm_classifier.fit(feature_matrix_train, target_train)  



