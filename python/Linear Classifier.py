
import numpy as np # linear algebra shit
from sklearn import preprocessing, neighbors 
from sklearn.model_selection import train_test_split
'''
preprocessing is for cleaning/scaling data (not used here)
train_test_split is for seperating training and texting data
neighbors is our classifier
'''
import pandas as pd # processes data
import pickle # if you want to save the classifier without training again


# use pandas to parse input
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# replace unkown data with outliers, I could have dropped the unkowns but sometimes your data has so many unkowns
# that dropping it will leave you with a much smaller data set
df.replace('?',-99999, inplace=True)

# irrelevent feature, does not affect class, improves accuracy if we remove
df.drop(['id'], 1, inplace=True)

# init features
X = np.array(df.drop(['class'], 1))
# init classes
y = np.array(df['class'])

# seperate training & testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define classifier
clf = neighbors.KNeighborsClassifier()

# train classifier
clf.fit(X_train, y_train)
# at this point we could pickle the classifier but since it runs so fast I didnt feel like it

# test
accuracy = clf.score(X_test, y_test)
print(accuracy)

# try our classifier out, in this case with two samples
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[3,2,1,5,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

