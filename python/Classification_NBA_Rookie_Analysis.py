
from __future__ import division
import graphlab
import math
import string
import random
import numpy
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


nba = graphlab.SFrame('nba_logreg1.csv')

train_data, test_data = nba.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

nba.head()

nba_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'target_5yrs',
                                                      features= ['gp','min','pts','fgm',
                                                                 'fga','fg','3p_made',
                                                                 '3pa','3p','ftm','fta','ft',
                                                                 'oreb','dreb',
                                                                 'reb','ast','stl',
                                                                 'blk','tov'],
                                                      validation_set=None)

nba_model

weights = nba_model.coefficients
weights.column_names()

num_positive_weights = (weights['value'] >=0).sum()
num_negative_weights = (weights['value']< 0).sum()

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

sample_test_data = test_data[11:14]
print sample_test_data['gp']
sample_test_data

print sample_test_data[1]['name']

print sample_test_data[0]['name']

scores = nba_model.predict(sample_test_data, output_type='margin')
print scores

def class_predications(scores):
    predications = []
    for score in scores:
        if score > 0:
            predication = 1
        else:
            predication = 0
        predications.append(predication)
    return predications

class_predications(scores)

print "Class predictions according to GraphLab Create:" 
print nba_model.predict(sample_test_data)

def calculate_probability(scores):
    probability_predictions = []
    for score in scores:
        probability_prediction = 1/(1+math.exp(-score))
        probability_predictions.append(probability_prediction)
    return probability_predictions

calculate_probability(scores)

print "Class predictions according to GraphLab Create:" 
print nba_model.predict(sample_test_data, output_type='probability')

test_data['probability_predictions'] = nba_model.predict(test_data,
                                                              output_type='probability')
test_data

test_data['name','probability_predictions'].topk('probability_predictions', k=20).print_rows(20)

test_data['name','probability_predictions'].topk('probability_predictions', k=20, reverse = True).print_rows(20)

print graphlab.SArray([1,1,1]) == sample_test_data['target_5yrs']
print nba_model.predict(sample_test_data) == sample_test_data['target_5yrs']

def get_classification_accuracy(model, data, true_labels):
    
    predicitions = model.predict(data)
    
   
    num_correct = sum(predicitions == true_labels)

 
    accuracy = num_correct/len(data)
    
    return accuracy

get_classification_accuracy(nba_model, test_data, test_data['target_5yrs'])

num_positive  = (train_data['target_5yrs'] == +1).sum()
num_negative = (train_data['target_5yrs'] == -0).sum()
print num_positive
print num_negative

print (test_data['target_5yrs'] == +1).sum()
print (test_data['target_5yrs'] == -0).sum()

print (test_data['target_5yrs'] == +1).sum()/len(test_data['target_5yrs'])

