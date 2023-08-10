import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn import preprocessing

titanic_data = pd.read_csv("./Datasets/titanic_train.csv")
titanic_data.head(20)

titanic_data = titanic_data.drop(['PassengerId','Name', 'Ticket', 'Cabin' ], axis=1)

titanic_data['Sex'] = titanic_data['Sex'].map({'female': 1, 'male': 0})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

titanic_data.head(10)

# For now lets drop thos rows with missing values ):
titanic_data = titanic_data.dropna()

dataMatrix = titanic_data.as_matrix()

NbPoints = titanic_data.shape[0]
NbVariables = titanic_data.shape[1] #because the result is not considered a variable, but we added a new clm with 1's

X = np.delete(dataMatrix, 0, 1) #only the parameters
X = np.insert(X, 0, 1, axis = 1) #Add a column with all 1 for independen parameter

Y = dataMatrix.T[0].reshape(NbPoints, 1) # Only the result which is the nb or rings as a column

# Define hypothesis function

# Th is the current Theta vector for our linear model
# X is the matrix with the training data
# i is the row of the data we want to know the prediction of 
def hyp (Th, X_i): 
    return 1 / (1 + ( np.exp( - np.dot( Th, X_i ) ) ) )

# Define the gradient correspondent to the logistic const function

# Th is the previous value of Th we had as a vector containing many thetas
# j is the index of Th we wish to upate, we will have to update all or some for stochastic gradient descent
# X is a matrix with colums being the variables used for learning
# Y is a column vecotr which gives the correct results for the parameters of X
def gradient (Th, j, X, Y):
    gradient = 0
    
    for i in range(NbPoints):
        gradient += ( hyp( Th, X[i] ) - Y[i] ) * X[i][j]
    
    return gradient

# Use the gradient of each parameter to update accordingly
Th = np.ones(NbVariables) # Initialize to random values like all 1's
Alph = 0.001
m = NbPoints

iterations = 70

### for printing the steps ###
lables = ['k', 'indep','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
t = PrettyTable(lables)

for k in range(iterations):
    NewTh = Th #Define a new th as placeholder because we want to update all at once
    
    for j in range(NbVariables):
        NewTh[j] -= Alph/m * gradient(Th, j, X, Y)
    
    Th = NewTh
    t.add_row(np.insert( np.round(Th, decimals=3), 0, k+1 ) ) # Add to the printer table, prepend the step
    
print(t)

# Use the gradient of each parameter to update accordingly
Th = np.array([ 0.784,  0.143 ,  1.035, -0.082,  0.629, 0.721,  0.005,  0.923]) # Pre rendered values
Alph = 0.001
m = NbPoints #should be nb of points in set

iterations = 1000
memo = np.array([Th]) # Lets give the CPU a brake

for k in range(iterations):
    NewTh = Th #Define a new th as placeholder because we want to update all at once
    
    for j in range(NbVariables):
        NewTh[j] -= Alph/m * gradient(Th, j, X, Y)
    
    Th = NewTh
    if (k % 100 == 0):
        memo = np.append(memo ,[Th], axis=0)

t = PrettyTable(["Th_" + str(i) for i in range(NbVariables)])
for i in range(int(len(memo))):
    t.add_row( np.round(memo[i], 3) )
print(t)

def isAlive(Th, X, i):
    return 1. if hyp(X[i], Th) > .5 else 0.

t = PrettyTable(['% Chance', '-> exp', 'Real'])

for i in range(NbPoints):
    t.add_row( [ int(100*(hyp(Th, X[i]))), isAlive(Th, X, i) , Y[i][0]] )
    
print(t.get_string(start = 50, end = 100))

from sklearn import metrics

predictions = np.array([])
for i in range(NbPoints):
    predictions = np.append(predictions ,[isAlive(Th, X, i)])

print(str(int(100*metrics.accuracy_score(Y, predictions))) + "% accuracy on train data by us") #nice cast nesting btw

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, Y)
predictions = logreg.predict(X)
print(str(int(100*metrics.accuracy_score(Y.T[0], predictions))) + "% accuracy on train data by sklearn")



