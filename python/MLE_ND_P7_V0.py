from IPython.core.display import HTML
hide_code = ''
HTML('''
<style>
body {background-color: aliceblue;}
a {color: steelblue;}
h1, h2, h3 { color: #348ABD;}
</style>
<script>code_show = true; 
function code_display() {
    if (code_show) {
        $('div.input').each(function(id) {
            if (id == 0 || $(this).html().indexOf('hide_code') > -1) {$(this).hide();}
        });
        $('div.output_prompt').css('opacity', 0);
    } else {
        $('div.input').each(function(id) {$(this).show();});
        $('div.output_prompt').css('opacity', 1);
    }
    code_show = !code_show;
} 
$(document).ready(code_display);</script>
<form action="javascript: code_display()"><input \
style="color: steelblue; background: ghostwhite; opacity: 0.9; " \
type="submit" value="Click to display or hide code"></form>''')

hide_code
# Import libraries
import numpy as np
import pandas as pd
from time import time
import warnings
from IPython.display import display 

# Display for notebooks
get_ipython().magic('matplotlib inline')

################################
### ADD EXTRA LIBRARIES HERE ###
################################
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

hide_code
# Read student data
student_data = pd.read_csv("student-data.csv")
print ("Student data read successfully!")
student_data.describe().T

hide_code
# Calculate number of students
n_students = len(student_data)

# Calculate number of features
n_features = len(list(student_data.T.index))

# Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])

# Calculate failing students
n_failed = len(student_data[student_data['passed'] == 'no'])

# Calculate graduation rate
grad_rate = n_passed * 100.0 / n_students

# Print the results
print ("Total number of students: {}".format(n_students))
print ("Number of features: {}".format(n_features))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))
print ("Graduation rate of the class: {:.2f}%".format(grad_rate))

hide_code
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print ("\nFeature values:")
print (X_all.head(7).T)

hide_code
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".       format(len(X_all.columns), list(X_all.columns)))

hide_code
# Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=1.0*num_test/len(X_all), random_state=1)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

hide_code
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))
    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".           format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

# Import the three supervised learning models from sklearn

# Initialize the three models
clf_A = AdaBoostClassifier()
clf_B = GradientBoostingClassifier()
clf_C = RandomForestClassifier()

# Set up the training set sizes
X_train_100, y_train_100 = X_train[:100], y_train[:100]
X_train_200, y_train_200 = X_train[:200], y_train[:200]
X_train_300, y_train_300 = X_train, y_train

# Execute the 'train_predict' function for each classifier and each training set size
for clf in [clf_A, clf_B, clf_C]:
    for (X_train, y_train) in [(X_train_100, y_train_100), 
                               (X_train_200, y_train_200), 
                               (X_train_300, y_train_300)]:
        train_predict(clf, X_train, y_train, X_test, y_test)

# Import 'GridSearchCV' and 'make_scorer'

# Create the parameters list you wish to tune
parameters = {'n_estimators': [50,150,250], 
              'learning_rate':[0.1,0.5,1.0]}

# Initialize the classifier
clf = AdaBoostClassifier()

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring=f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print ("Tuned model has a training F1 score of {:.4f}.".       format(predict_labels(best_clf, X_train, y_train)))
print ("Tuned model has a testing F1 score of {:.4f}.".       format(predict_labels(best_clf, X_test, y_test)))
print ("Tuned model has the parameters: \n{}".format(best_clf.get_params()))

