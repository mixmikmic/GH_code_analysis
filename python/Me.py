#Importing all essential modules

#pandas
import pandas as pd
from pandas import Series,DataFrame

#numpy, matplotlib,seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
get_ipython().magic('matplotlib inline')

#machine learning modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#get data for training and testing
titanic_df = pd.read_csv("/home/satish/Documents/A_Journey_through_Titanic_kaggle_Data/train.csv")
test_df = pd.read_csv("/home/satish/Documents/A_Journey_through_Titanic_kaggle_Data/test.csv")

NumberOfPassengers = len(test_df) + len(titanic_df)
print "# of Examples:",NumberOfPassengers

NumberOfFeatures = len(titanic_df.columns)
print "# of features:",NumberOfFeatures



