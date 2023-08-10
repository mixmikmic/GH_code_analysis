#First import all the libraries needed

import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt #for plotting the graphs
from sklearn.linear_model import LogisticRegression #for logistic regression
from sklearn.pipeline import Pipeline #to assemble steps for cross validation
from sklearn.preprocessing import PolynomialFeatures #for all the polynomial features
from sklearn import svm #for Support Vector Machines
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
from sklearn.naive_bayes import GaussianNB  #for naive bayes classifier
from scipy import stats #for statistical info

from sklearn.cross_validation import train_test_split # to split the data in train and test
from sklearn.cross_validation import KFold # for cross validation
from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn.neighbors import KNeighborsClassifier  #for k-neighbor classifier
from sklearn import metrics  # for checking the accuracy 
from time import time

#load data
data = pd.read_csv("data.csv")

#check out the first two rows to make sure it loaded correctly
data.head(2)

#Description of the dataset

#how many cases are included in the dataset
length = len(data)
#how many features are in the dataset
features = data.shape[1]-1

# Number of malignant cases
malignant = len(data[data['diagnosis']=='M'])

#Number of benign cases
benign = len(data[data['diagnosis']=='B'])

#Rate of malignant tumors over all cases
rate = (float(malignant)/(length))*100

print "There are "+ str(len(data))+" cases in this dataset"
print "There are {}".format(features)+" features in this dataset"
print "There are {}".format(malignant)+" cases diagnosed as malignant tumor"
print "There are {}".format(benign)+" cases diagnosed as benign tumor"
print "The percentage of malignant cases is: {:.4f}%".format(rate)

data.diagnosis.unique()

#make diagnosis column numerical
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})


data.head(10)

#explore further
data.describe()

#drop ID because we do not need the ID number as shown above

data.drop('id',axis=1,inplace=True)

#check that id is dropped
data.head(10)

#describe again
data.describe()

#explore some of the attribute for best understanding

#radius mean is the mean of distances from center to points on the perimeter of breast mass
#so let's look at maximum, minimum, average, and standard deviation of radius mean
min_radius = min(data['radius_mean'])
max_radius = max(data['radius_mean'])
average_radius = np.mean(data['radius_mean'])
sd_radius = np.std(data['radius_mean'])

print "Minimum of radius mean is: {:,.2f} ".format(min_radius)
print "Maximum of radius mean is: {:,.2f} ".format(max_radius)
print "Average of radius mean is: {:,.2f} ".format(average_radius)+"with a standard deviation of {:,.2f}".format(sd_radius)

#texture mean is the standard deviation of gray scale value
#so let's look at maximum, minimum, average, and standard deviation of texture mean
min_texture = min(data['texture_mean'])
max_texture = max(data['texture_mean'])
average_texture = np.mean(data['texture_mean'])
sd_texture = np.std(data['texture_mean'])

print "Minimum of texture mean is: {:,.2f} ".format(min_texture)
print "Maximum of texture mean is: {:,.2f} ".format(max_texture)
print "Average of texture mean is: {:,.2f} ".format(average_texture)+"with a standard deviation of {:,.2f}".format(sd_texture)


def histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=20,ax=ax)
        ax.set_title(" Distribution of "+ var_name)
    #fig.tight_layout()
    plt.rcParams.update({'font.size': 6, 'font.weight':'bold'})
    plt.show()

features=list(data.columns[1:5])
histograms(data,features,2,2)

features1=list(data.columns[5:9])
histograms(data,features1,2,2)

features2=list(data.columns[9:11])
histograms(data,features2,1,2)

#to see how distribution is in regard to the diagnosis, we need to first split
#the dataset into two groups
malignant = data[data['diagnosis'] ==1]
benign = data[data['diagnosis'] ==0]

#also bring features back, basically redefining them again- the first ten

features = list(data.columns[1:11])

#just to check that this works
malignant[features].head(2)

plt.rcParams.update({'font.size': 8, 'font.weight':'bold', 'font.family':'italics'})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for x,y in enumerate(axes):
    y.figure
    binwidth= (max(data[features[x]]) - min(data[features[x]]))/30
    y.hist([malignant[features[x]],benign[features[x]]], bins=np.arange(min(data[features[x]]), max(data[features[x]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['Malignant','Benign'],color=['silver','orangered'])
    y.legend(loc='upper right')
    y.set_title(features[x])
plt.tight_layout()
plt.show()



