import numpy as np 
from math import sqrt
import pandas as pd
import random
from statistics import mode
import os
print("All dependencies successfully imported.")

os.chdir('/Users/Sam/Documents/Python/Datasets')

#Reading in the data set to a dataframe
orig_data = pd.read_csv('breast_cancer.txt')

#Next, separate the data into the features (predictors) and labels (classes: malignant or benign)
data = orig_data.drop(['class', 'patient_id'], axis = 1)
label = orig_data['class']

#Now we need to replace missing values in the data with the number zero.
data['bare_nuclei'].replace('?',0.0, inplace = True)
data['bare_nuclei'] = data['bare_nuclei'].astype('float')

#Convert both features (data) and labels
data = list(data.values)
label = list(label)

#Reference dictionary for easy to read results. 
dict = {2:'benign',4:'malignant'}

#Must sort from smallest to largest to get similarity rankings.
def euclid(x,y):
    sums = 0.0
    for i in range(len(x)):
        sums += pow((x[i] - y[i]),2)

    return sqrt(sums)

#Split data to train and test samples
#The random.random() function decides whether an item goes into the train or test set
#train_count and test_count keep track of how many records are in each sample
#x_traintest acts as a reference bank for all testing data later. It will be what "unknown" ...
#data points are compared against
def train_test_split(data,label, test_ratio):
    x_traintest = []
    x_test = []
    y_test = []
    train_count = 0
    test_count = 0
    for i in range(len(data)):
        if random.random() < test_ratio:
            test_count += 1
            x_test.append(data[i])
            y_test.append(label[i])
        else:
            train_count += 1
            x_traintest.append((data[i],label[i]))
            
    return x_traintest, x_test, y_test

# For each record in the training dataset, this determines a euclidian score
# These scores are sorted, and the labels  of the most similar k values
# Are returned (e.g. (5, Benign), (13.5, Malignant), etc.)

def getNeighbors(data,testInstance,similarity,k):
    neighbor = []
    for i in range(len(data)):
        neighbor.append(((similarity(data[i][0],testInstance)),data[i][1]))
    
    neighbor.sort(key=lambda tup: tup[0]) 
    
    neighbors = pd.DataFrame()
    neighbors['euclid'] = [x[0] for x in neighbor]
    neighbors['label'] = [x[1] for x in neighbor]
    
    return neighbors['label'][:k].tolist()

def get_results(x_traintest,x_test,k):
    res_list = []
    label = [x[1] for x in x_data]
    for i in range(len(x_test)):
        kNeighbor = getNeighbors(x_traintest, x_test[i],euclid,k)
        res_list.append(mode(kNeighbor))
    return res_list

def view_results(actual,predicted):
    result = pd.DataFrame()
    for i in range(len(actual)):
        actual[i] = dict[actual[i]]
        predicted[i] = dict[predicted[i]]
    result['ACTUAL'] = actual
    result['PREDICTED'] = predicted
    result['CORRECT?'] = [actual[i] == predicted[i] for i in range(len(actual))]
    
    return result

def get_accuracy(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    num_correct = count
    num_incorrect = len(actual) - count
    return ("Accuracy: " + str(count / len(actual)*100) + " percent" + '\n' + 
           "Number correct: " + str(num_correct) + '\n' + 
           "Number incorrect: " + str(num_incorrect))

#Makes testing data sample 20% of the entire dataset. This can be adjusted.
#For the purposes of this model, this is the training step.
x_data, x_test, y_test = train_test_split(data,label,0.2)

#This tests the model and returns a list of results
predicted_results = get_results(x_data,x_test,5)

#Organizing our results so we can see 20 of our test cases
viewed_results = view_results(y_test,predicted_results)
viewed_results = viewed_results[:20]

#This allows us to see the results
print(viewed_results)
print()
print(get_accuracy(y_test,predicted_results))



