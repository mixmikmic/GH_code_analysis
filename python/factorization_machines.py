get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sns

class FactorizationMachines:
    def __init__(
            self,
            lambda0=1,
            lambda1=1,
            lambda2=1,
            n_iter=100,
            learning_rate=1,
            k=50,
            random_seed=None
    ):
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.k = k
        self.random_seed = random_seed
        if (self.random_seed is not None):
            self.random_seed = int(self.random_seed)
    
    def __updateParametersMiniBatch(self, X,y,B):
        y_hat = self.__predict(X)
        y_hat = np.minimum(self.maxY, y_hat)
        y_hat = np.maximum(self.minY, y_hat)
        term1 = 2*(y_hat - y[:,np.newaxis])
        gradW0 = np.sum(term1, axis = 0) + self.lambda0*self.w0
        if(gradW0.shape[0] <> 1 or gradW0.shape[1] <> 1):
            raise ValueError('Gradient W0 has more than 1 element')
        gradW = np.sum(np.multiply(term1,X), axis = 0) + self.lambda1*self.w
        if(gradW.shape[1] <> X.shape[1]):
            raise ValueError('Gradient W has incorrect number of elements')   
        # Alternate implementation instead of the for loop over instances.
        #gradV = np.multiply(term1,X).T*(X*self.V)
        #term2 = np.array(map(lambda row: np.multiply(self.V,row.T), np.multiply(term1,np.square(X))))
        #term2 = np.sum(term2,axis=0)
        #gradV -= term2
        gradV = np.zeros(self.V.shape)
        D = X.shape[0]
        instances = range(0,D)
        for d in instances:
            xd = X[d,:]
            xdV = xd*self.V
            gradV += term1[d][0,0] * (
                        np.multiply(xd.T,np.repeat(xdV,xd.shape[1],axis=0)) -
                        np.multiply(self.V,np.square(xd.T))
                    ) 
        
        gradV = gradV + self.lambda2*self.V
        self.w0 -= (self.learning_rate/B*1.0)*gradW0;
        self.w  -= (self.learning_rate/B*1.0)*gradW;
        self.V  -= (self.learning_rate/B*1.0)*gradV
        
    def __updateParameters(self, X,y):
        y_hat = self.__predict(X)
        y_hat = np.minimum(self.maxY, y_hat)
        y_hat = np.maximum(self.minY, y_hat)
        term1 = 2*(y_hat - y)
        gradW0 = term1 + self.lambda0*self.w0
        if(gradW0.shape[0] <> 1 or gradW0.shape[1] <> 1):
            raise ValueError('Gradient W0 has more than 1 element')
        gradW = np.multiply(term1,X) + self.lambda1*self.w
        if(gradW.shape[1] <> X.shape[1]):
            raise ValueError('Gradient W has incorrect number of elements')   
        gradV = np.zeros(self.V.shape)
        xdV = X*self.V
        gradV += term1[0,0] * (
                    np.multiply(X.T,np.repeat(xdV,X.shape[1],axis=0)) -
                    np.multiply(self.V,np.square(X.T))
                ) 
        gradV += self.lambda2*self.V
        self.w0 -= self.learning_rate*gradW0;
        self.w  -= self.learning_rate*gradW;
        self.V  -= self.learning_rate*gradV
    
    def __predict(self,X):
        return (self.w0 + 
                X*self.w.T + 
                0.5*(np.sum(np.square(X*self.V) - (np.square(X)*np.square(self.V)),axis=1))
               )
    
    def fit(self,X,y):
        if type(X) != np.matrix:
            X = np.matrix(X)
        if type(y) != np.array:
            y = np.array(y)
        if (len(X.shape)!= 2):
            raise ValueError('X should be a 2-D matrix')
        if (X.shape[0] != len(y)):
            raise ValueError('X and y should contain the same number of examples')
        D = X.shape[0]
        n = X.shape[1]
        self.minY = min(y)
        self.maxY = max(y)
        self.w0 = 0
        self.w = np.zeros([1,n])
        self.V = np.random.normal(scale=0.1,size=(n, self.k))
        indx = range(0,D)
        minibatch_size = 500
        if (self.random_seed is not None):
            np.random.seed(self.random_seed)
        if (D >= 1000):
            print("Minibatch size: " + str(minibatch_size))
            for i in range(0,self.n_iter):
                print("Epoch number: " + str(i))
                shuffledIndx = np.random.permutation(indx)
                start = 0;
                end = minibatch_size
                while (start < D):
                    xbatch = X[shuffledIndx[start:end],:]
                    ybatch = y[shuffledIndx[start:end]]
                    self.__updateParametersMiniBatch(xbatch,ybatch,xbatch.shape[0])
                    start += minibatch_size
                    end += minibatch_size
                    if (end > D):
                        end = D
        else:
            for i in range(0, self.n_iter):
                print("Epoch number: " + str(i))
                shuffledIndx = np.random.permutation(indx)
                for d in shuffledIndx:
                    xd = X[d,:]
                    yd = y[d]
                    self.__updateParameters(xd,yd)

    def predict(self,X,y=None):
        if type(X) != np.matrix:
            X = np.matrix(X)
        if type(y) != np.array:
            y = np.array(y)
        y_hat = self.__predict(X)
        y_hat = np.minimum(self.maxY, y_hat)
        y_hat = np.maximum(self.minY, y_hat)
        return y_hat

from sklearn.feature_extraction import DictVectorizer

# Read in data
def loadData(filename,path="/Users/gmuralidhar/Projects/MLDatasets/movielens-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")
v = DictVectorizer()
X_train = v.fit_transform(train_data).toarray()
X_test = v.transform(test_data).toarray()
print X_train.shape
print X_test.shape
print len(train_users)
print len(train_items)

f = FactorizationMachines(lambda0 = 0.01, lambda1 = 0.01, lambda2 = 0.01, k = 10, n_iter = 100, learning_rate = 0.5)
f.fit(X_train,y_train)

print("-------------------------------------------------- w0 -------------------------------------------------") 
print ("\n")
print f.w0
print ("\n")
print("-------------------------------------------------- w --------------------------------------------------") 
print ("\n")
print f.w
print ("\n")
print("-------------------------------------------------- V --------------------------------------------------") 
print ("\n")
print f.V
print ("\n")
print np.reshape(f.V,(1,26230))

print ("Number of test examples = " + str(X_test.shape[0]))
y_hat = f.predict(np.matrix(X_test))
print ("Number of predicted examples = " + str(y_hat.shape[0]))
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_hat)
print ("Mean squared error = " + str(mse))

