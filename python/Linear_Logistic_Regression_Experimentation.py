import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import scipy.optimize as opt 
import random 
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

def gradientDescent_LinearRegression(X,y,theta, alpha, iterations):
    m = len(X)
    temp = np.matrix(np.zeros(theta.shape))
    params = int(theta.shape[1])
    cost = np.zeros(iters)
        
    for i in range(iters):
        error = (X*theta.T) - y
        
        for j in range(params):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (1/m)*alpha*np.sum(term)
            
        theta = temp
        cost[i] = costFunction_LinearRegression(X,y,theta)
  
    return theta , cost

def costFunction_LinearRegression(X_train,y_train,theta):
    m = len(y_train) 
    J = np.sum(np.power(((X_train*theta.T) - y_train),2))/(2*m)
    return J


def data_preparation(data,colNos):
    
    # slicing non-predictive attributes
    #if('url' in data.columns and 'timedelta' in data.columns):
    data = data.drop(["url"," timedelta"], axis=1)
    
    if colNos == "All":
        pass
    else:
        colNos.append(58) # Adding last coulumn "shares" 
        data = data[data.columns[colNos]]  
        
    cols = data.shape[1]  
    train, test = train_test_split(data, test_size=0.3,random_state=0)

    X_train=train.iloc[:,0:cols-1]
    y_train=train.iloc[:,cols-1:cols]

    X_test=test.iloc[:,0:cols-1]
    y_test=test.iloc[:,cols-1:cols]

    #Normalization
    X_train = (X_train - X_train.mean())/ (X_train.max() - X_train.min())
    X_test = (X_test - X_test.mean())/ (X_test.max() - X_test.min())

    X_train.insert(0, 'X0', 1)
    X_test.insert(0, 'X0', 1)
 
    X_train = np.matrix(X_train.values)  
    X_test = np.matrix(X_test.values)  
    y_train = np.matrix(y_train.values)  
    y_test = np.matrix(y_test.values)  
    theta = np.matrix(np.zeros([1,X_train.shape[1]]))

    return X_train,X_test,y_train,y_test,theta

def results_Linear(theta,X,y,data_set):
    y_predictions = X*OptTheta.T
    print("____________________________________________________")
    print("Linear Regression | Running the algo on "+data_set)
    print("Mean squared error = {}".format(round(mean_squared_error(y,y_predictions),4)))    
    print("Mean Absolute error = {}".format(round(mean_absolute_error(y,y_predictions),4)))
    print("____________________________________________________")


# Linear Regression

data = pd.read_csv("onlineNewsPopularity.csv")

X_train,X_test,y_train,y_test,theta = data_preparation(data,"All")

alpha = 0.05
iters = 500
OptTheta, cost = gradientDescent_LinearRegression(X_train, y_train, theta, alpha, iters)  
y_predictions = X_test*OptTheta.T

results_Linear(theta,X_train,y_train,"Train Data")
results_Linear(theta,X_test,y_test,"Test Data")


def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def costFunction_LogisticRegression(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum((first - second)/len(X))

def gradientDescent_LogisticRegression(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    params = int(theta.ravel().shape[1])
    grad = np.zeros(params)
    
    error = sigmoid(X*theta.T) - y
    
    for i in range(params):
        term = np.multiply(error,X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

def predictions_logistic(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x >=0.5 else 0 for x in probability]

def results_Logistic(theta,X,y,data_set):
    y_predictions = np.matrix(predictions_logistic(theta,X)).T
    temp = (y_predictions == y)
    accuracy = (temp.sum()/temp.shape[0])*100
    print("____________________________________________________")
    print("Logistic Regression | Running the algo on "+data_set)
    print("Accuracy is {} %".format(round(accuracy,2)))
    print("Confusion Matrix = \n{}".format(confusion_matrix(y,y_predictions)))    
    print("Mean Absolute error = {}".format(round(mean_absolute_error(y,y_predictions),4)))
    print("____________________________________________________")

#Logistic regression

data = pd.read_csv("onlineNewsPopularity.csv")

median = data[' shares'].median()
data[' shares'] = np.where(data[' shares']>=median, 1, 0)

# 1: represents Large
# 0: represents Small

X_train,X_test,y_train,y_test,theta = data_preparation(data,"All")

result = opt.fmin_tnc(func=costFunction_LogisticRegression, x0=theta, fprime=gradientDescent_LogisticRegression, args=(X_train, y_train))  
OptTheta = np.matrix(result[0])

#cost_logistic_regression(OptTheta, X_train, y_train)  

results_Logistic(OptTheta,X_test,y_test,"Test set")
results_Logistic(OptTheta,X_train,y_train,"Train set")

# Linear Regression

data = pd.read_csv("onlineNewsPopularity.csv")

X_train,X_test,y_train,y_test,theta = data_preparation(data,"All")
  
iters = 500
fig, ax = plt.subplots(figsize=(20,20))  
sample=[1,2,3,4,5]
count = 510
for alpha in [0.03,0.1,0.3,1,3]:
    count+=1
    OptTheta, cost = gradientDescent_LinearRegression(X_train, y_train, theta, alpha, iters)  
    sample[count-511] = cost
    plt.subplot(count)
    plt.plot(np.arange(iters), sample[count-511], 'b--')
    plt.title('Cost Vs Iterations where alpha = {}'.format(alpha))
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.tight_layout()

# Linear Regression

data = pd.read_csv("onlineNewsPopularity.csv")

X_train,X_test,y_train,y_test,theta = data_preparation(data,random.sample(range(0,58),10))

alpha = 0.05
iters = 500
OptTheta, cost = gradientDescent_LinearRegression(X_train, y_train, theta, alpha, iters)  

#print(OptTheta)
#print(costFunction(X_train,y_train,OptTheta))

results_Linear(OptTheta,X_train,y_train,"Train data")
results_Linear(OptTheta,X_test,y_test,"Test data")

# Linear Regression

data = pd.read_csv("onlineNewsPopularity.csv")

data = data[['url',' timedelta',' is_weekend',' n_tokens_title',' num_imgs',' num_videos',' global_subjectivity',' global_sentiment_polarity',' global_rate_positive_words',' title_subjectivity',' title_sentiment_polarity',' abs_title_subjectivity',' shares']]

X_train,X_test,y_train,y_test,theta = data_preparation(data,"All")

alpha = 0.05
iters = 500
OptTheta, cost = gradientDescent_LinearRegression(X_train, y_train, theta, alpha, iters)  

results_Linear(theta,X_train,y_train,"Train Data")
results_Linear(theta,X_test,y_test,"Test Data")

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset

# Linear Regression

data = pd.read_csv("onlineNewsPopularity.csv")
data = correlation(data,0.75)

X_train,X_test,y_train,y_test,theta = data_preparation(data,"All")

alpha = 0.05
iters = 500
OptTheta, cost = gradientDescent_LinearRegression(X_train, y_train, theta, alpha, iters)  

results_Linear(theta,X_train,y_train,"Train Data")
results_Linear(theta,X_test,y_test,"Test Data")

