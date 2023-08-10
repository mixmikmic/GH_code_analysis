# Import libraries
import numpy as np
from rlscore.learner import RLS
from rlscore.measure import sqerror
from rlscore.learner import LeaveOneOutRLS

# Function to load dataset and split in train and test sets
def load_housing():
    np.random.seed(1)
    D = np.loadtxt("/Volumes/PANZER/Github/learning-space/Datasets/02 - Classification/housing_data.txt")
    np.random.shuffle(D)
    X = D[:,:-1] # Independent variables
    Y = D[:,-1]  # Dependent variable
    X_train = X[:250]
    Y_train = Y[:250]
    X_test = X[250:]
    Y_test = Y[250:]
    return X_train, Y_train, X_test, Y_test

def print_stats():
    X_train, Y_train, X_test, Y_test = load_housing()
    print("Housing data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Test set: %d instances, %d features" %X_test.shape)

if __name__ == "__main__":
    print_stats()

# Function to train RLS method
def train_rls():
    #Trains RLS with default parameters (regparam=1.0, kernel='LinearKernel')
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = RLS(X_train, Y_train)
    
    #Leave-one-out cross-validation predictions, this is fast due to
    #computational short-cut
    P_loo = learner.leave_one_out()
    
    #Test set predictions
    P_test = learner.predict(X_test)
    
    # Stats
    print("leave-one-out error %f" %sqerror(Y_train, P_loo))
    print("test error %f" %sqerror(Y_test, P_test))
    
    #Sanity check, can we do better than predicting mean of training labels?
    print("mean predictor %f" %sqerror(Y_test, np.ones(Y_test.shape)*np.mean(Y_train)))

if __name__=="__main__":
    train_rls()

def train_rls():
    #Select regparam with leave-one-out cross-validation
    X_train, Y_train, X_test, Y_test = load_housing()
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_error = float("inf")
   
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        
        #Leave-one-out cross-validation predictions, this is fast due to
        #computational short-cut
        P_loo = learner.leave_one_out()
        e = sqerror(Y_train, P_loo)
        print("regparam 2**%d, loo-error %f" %(log_regparam, e))
        if e < best_error:
            best_error = e
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    print("best regparam %f with loo-error %f" %(best_regparam, best_error)) 
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()

def train_rls():
    #Trains RLS with automatically selected regularization parameter
    X_train, Y_train, X_test, Y_test = load_housing()
    
    # Grid search
    regparams = [2.**i for i in range(-15, 16)]
    learner = LeaveOneOutRLS(X_train, Y_train, regparams = regparams)
    loo_errors = learner.cv_performances
    P_test = learner.predict(X_test)
    print("leave-one-out errors " +str(loo_errors))
    print("chosen regparam %f" %learner.regparam)
    print("test error %f" %sqerror(Y_test, P_test))

if __name__=="__main__":
    train_rls()

def train_rls():
    #Selects both the gamma parameter for Gaussian kernel, and regparam with loocv
    X_train, Y_train, X_test, Y_test = load_housing()
    
    regparams = [2.**i for i in range(-15, 16)]
    gammas = regparams
    best_regparam = None
    best_gamma = None
    best_error = float("inf")
    
    for gamma in gammas:
        #New RLS is initialized for each kernel parameter
        learner = RLS(X_train, Y_train, kernel="GaussianKernel", gamma=gamma)
        for regparam in regparams:
            #RLS is re-trained with the new regparam, this
            #is very fast due to computational short-cut
            learner.solve(regparam)
            
            #Leave-one-out cross-validation predictions, this is fast due to
            #computational short-cut
            P_loo = learner.leave_one_out()
            e = sqerror(Y_train, P_loo)
            
            #print "regparam", regparam, "gamma", gamma, "loo-error", e
            if e < best_error:
                best_error = e
                best_regparam = regparam
                best_gamma = gamma
    learner = RLS(X_train, Y_train, regparam = best_regparam, kernel="GaussianKernel", gamma=best_gamma)
    P_test = learner.predict(X_test)
    print("best parameters gamma %f regparam %f" %(best_gamma, best_regparam))
    print("best leave-one-out error %f" %best_error)
    print("test error %f" %sqerror(Y_test, P_test))
    
    
if __name__=="__main__":
    train_rls()    

from rlscore.utilities.reader import read_svmlight

# Load dataset and stats
def print_stats():
    X_train, Y_train, foo = read_svmlight("/Volumes/PANZER/Github/learning-space/Datasets/02 - Classification/a1a.t")
    X_test, Y_test, foo = read_svmlight("/Volumes/PANZER/Github/learning-space/Datasets/02 - Classification/a1a")
    print("Adult data set characteristics")
    print("Training set: %d instances, %d features" %X_train.shape)
    print("Test set: %d instances, %d features" %X_test.shape)

if __name__=="__main__":
    print_stats()

from rlscore.learner import RLS
from rlscore.measure import accuracy
from rlscore.utilities.reader import read_svmlight


def train_rls():
    # Train ans test datasets    
    X_train, Y_train, foo = read_svmlight("/Volumes/PANZER/Github/learning-space/Datasets/02 - Classification/a1a.t")
    X_test, Y_test, foo = read_svmlight("/Volumes/PANZER/Github/learning-space/Datasets/02 - Classification/a1a", X_train.shape[1])
    learner = RLS(X_train, Y_train)
    best_regparam = None
    best_accuracy = 0.
    
    #exponential grid of possible regparam values
    log_regparams = range(-15, 16)
    for log_regparam in log_regparams:
        regparam = 2.**log_regparam
        #RLS is re-trained with the new regparam, this
        #is very fast due to computational short-cut
        learner.solve(regparam)
        
        #Leave-one-out cross-validation predictions, this is fast due to
        #computational short-cut
        P_loo = learner.leave_one_out()
        acc = accuracy(Y_train, P_loo)
        
        print("regparam 2**%d, loo-accuracy %f" %(log_regparam, acc))
        if acc > best_accuracy:
            best_accuracy = acc
            best_regparam = regparam
    learner.solve(best_regparam)
    P_test = learner.predict(X_test)
    
    print("best regparam %f with loo-accuracy %f" %(best_regparam, best_accuracy)) 
    print("test set accuracy %f" %accuracy(Y_test, P_test))

if __name__=="__main__":
    train_rls()

