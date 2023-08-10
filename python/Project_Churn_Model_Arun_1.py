import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
get_ipython().magic('matplotlib inline')
import seaborn as sns
import graphlab as gl
gl.canvas.set_target('ipynb')

churn_data = pd.read_csv('./churn.csv')

# Looking at the data characteristics
churn_data.info()

churn_data.head()

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_data[yes_no_cols] = churn_data[yes_no_cols] == 'yes'

churn = ['Churn?']
churn_data[churn] = churn_data[churn] == 'True.'

# converting boolean columns to integer type

Intl_plan = churn_data["Int'l Plan"]
churn_data["Int'l Plan"] = Intl_plan.astype(int)

vMail_plan = churn_data["VMail Plan"]
churn_data["VMail Plan"] = vMail_plan.astype(int)

churn_new = churn_data['Churn?']
churn_data['Churn'] = churn_new.astype(int)

# dropping the irrelevant data
#churn_data = churn_data.drop(['State','Area Code','Phone','Churn?'],axis=1)

# calculating the monthly bill
churn_data['month_bill'] = churn_data['Day Charge']+churn_data['Eve Charge']+churn_data['Night Charge']+churn_data['Intl Charge']

churn_data.info()

churn_data.head()

churn_data.describe()

pd.scatter_matrix(churn_data, figsize=(17, 10), edgecolor='none', alpha=0.5);

churn_data.corr()


fig, ax = plt.subplots(figsize=(60, 60))
sns.corrplot(churn_data,ax=ax)

# Remove charge features as they're correlated with minutes
# Drop area code feature, phone, 

churn_count=pd.value_counts(churn_data['Churn'])
churn_count.plot(kind='bar')

churn_count = pd.value_counts(churn_data['CustServ Calls'])
churn_count.plot(kind='bar')

churn_data.hist('Account Length',bins = 15);

churn_data.hist('month_bill',bins = 10);

#importing the data into graphlab
churn_sFrame = gl.SFrame(churn_data)

churn_sFrame.show()

churn_sFrame.head()

churn_sFrame.show(view = 'Bar Chart', x = 'Account Length',y= 'Churn')

churn_sFrame.show(view = 'Bar Chart', x = 'CustServ Calls',y= 'Churn')

churn_sFrame.rename({"Int'l Plan": 'Intl Plan'})

train_data,test_data = churn_sFrame.random_split(.8, seed=0)

# features_set = ['Account Length',"Int'l Plan", 'VMail Plan',  'VMail Message', 'Day Mins', 'Day Calls',
#                 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 
#                 'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls']

features_set = ['State', 'Account Length', 'Intl Plan','VMail Plan', 'VMail Message', 'Day Mins', 'Day Calls', 
                 'Eve Mins', 'Eve Calls', 'Night Mins','Night Calls', 'Intl Mins', 'Intl Calls', 'CustServ Calls']

Labels = ['Churn']

#models = gl.classifier.create(train_data,target=Labels,features= features_set)

model_logistic = gl.logistic_classifier.create(train_data,
                                                     target='Churn',
                                                     features=features_set,
                                                     validation_set=test_data,
                                                     class_weights= 'auto',
                                                     l2_penalty=0.03,
                                                     max_iterations = 10)

y_pred=model_logistic.classify(train_data)
model_logistic.evaluate(test_data)
# Evaluate the model by making predictions of target values and comparing these to actual values.
result_model_logistic = model_logistic.evaluate(test_data)
gl.canvas.set_target('ipynb')
model_logistic.show(view='Evaluation')

# Create a (binary or multi-class) classifier model of type BoostedTreesClassifier using gradient boosted trees 
# Automatically take validation set for tuning
boosted_trees_model = gl.boosted_trees_classifier.create(train_data,
                                              features=features_set,
                                              class_weights= 'auto',           
                                              target = 'Churn')
# Return a classification, for each example in the dataset, using the trained logistic regression model.
predicitons = boosted_trees_model.classify(train_data)
# Evaluate the model by making predictions of target values and comparing these to actual values.
boosted_trees_model.evaluate(test_data)
boosted_trees_model.show(view='Evaluation')
# Evaluate the model by making predictions of target values and comparing these to actual values.
result_model_boosted = boosted_trees_model.evaluate(test_data)

# The prediction is based on a collection of base learners i.e decision tree classifiers
# Different from linear models like logistic regression or SVM, gradient boosted trees can model 
# non-linear interactions between the features and the target.
random_forest_model = gl.random_forest_classifier.create(train_data,
                                              features=features_set,
                                              class_weights= 'auto',
                                              target = 'Churn')

random_forest_model.classify(train_data)
random_forest_model.evaluate(test_data)
random_forest_model.show(view='Evaluation')
# Evaluate the model by making predictions of target values and comparing these to actual values.
result_model_random_forest = random_forest_model.evaluate(test_data)

# The prediction is based on a collection of base learners i.e decision tree classifiers
# Different from linear models like logistic regression or SVM, gradient boosted trees can model 
# non-linear interactions between the features and the target.
svm_model = gl.svm_classifier.create(train_data,
                                              features=features_set,
                                              class_weights= 'auto',
                                              target = 'Churn')

svm_model.classify(train_data)
svm_model.evaluate(test_data)
svm_model.show(view='Evaluation')
# Evaluate the model by making predictions of target values and comparing these to actual values.
svm_model_result = svm_model.evaluate(test_data)

def print_statistics(result):
    print "*" * 30
    print "Accuracy        : ", result["accuracy"]
    print "Precision       : ", result['precision']
    print "Recall          : ", result['recall']
    print "AUC             : ", result['auc']
    print "Confusion Matrix: \n", result["confusion_matrix"]

print_statistics(result_model_logistic)
print_statistics(result_model_boosted)
print_statistics(result_model_random_forest)
print_statistics(svm_model_result)

from graphlab import model_parameter_search
# Search over a grid of multiple hyper-parameters, with validation set
params = {'target': 'Churn'}

job = model_parameter_search.create((train_data,test_data),
                                        gl.boosted_trees_classifier.create,
                                        params)

results = job.get_results()
print results

# Create a (binary or multi-class) classifier model of type BoostedTreesClassifier using gradient boosted trees 
# Automatically take validation set for tuning
boosted_trees_model = gl.boosted_trees_classifier.create(train_data, 'Churn', features=features_set, max_iterations=100, 
                                   validation_set=test_data, class_weights= None, max_depth=4, step_size=0.1, 
                                   min_loss_reduction=1, min_child_weight=2, row_subsample=0.9, 
                                   column_subsample=1.0, verbose=True, random_seed=None)

# Return a classification, for each example in the dataset, using the trained logistic regression model.
predicitons = boosted_trees_model.classify(train_data)
# Evaluate the model by making predictions of target values and comparing these to actual values.
boosted_trees_model.evaluate(test_data)
boosted_trees_model.show(view='Evaluation')
# Evaluate the model by making predictions of target values and comparing these to actual values.
result_model_boosted = boosted_trees_model.evaluate(test_data)

print_statistics(result_model_boosted)

# Calculating the probabilities
predict_prob = boosted_trees_model.predict(churn_sFrame, output_type='probability')

churn_sFrame['predicted_prob']= predict_prob

churn_sFrame.head()

# Calculating the estimated loss
churn_sFrame['estimated_loss']= churn_sFrame['predicted_prob'] * churn_sFrame['month_bill']

churn_sFrame.print_rows(num_rows=100, num_columns=21)

# Look at lots of descriptive statistics of 'estimated_loss'
print "mean: " + str(churn_sFrame['estimated_loss'].mean())
print "std: " + str(churn_sFrame['estimated_loss'].std())
print "var: " + str(churn_sFrame['estimated_loss'].var())
print "min: " + str(churn_sFrame['estimated_loss'].min())
print "max: " + str(churn_sFrame['estimated_loss'].max())
print "sum: " + str(churn_sFrame['estimated_loss'].sum())



