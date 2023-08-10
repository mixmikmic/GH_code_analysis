import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

n = 3000  # number data points
d = 20    # dimension of input
depth = 3 # Base Classifier is a Decision Tree, deeper => "less weak". 
max_hypothesis = 500

X, y = make_classification(n, d, n_classes=2)

for i in range(10, max_hypothesis, 10):   
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),  algorithm="SAMME", n_estimators=i)
    ada.fit(X, y)
    
    # saved to "export/U[NUM HYPOTHESIS].txt"
    # and      "export/w[NUM HYPOTHESIS].txt"
    compute_and_save_matrix(ada, X, y) 
    
    print("\r%i/500"%i, end='')

def compute_and_save_matrix(ada, X, y):
    n, d = np.shape(X)
    
    # change from {0, 1} to {-1, 1}
    y = y*2-1
    
    # Normalize weights. 
    ada.estimator_weights_ = ada.estimator_weights_ / np.sum(ada.estimator_weights_)
    assert np.allclose(np.sum(ada.estimator_weights_), 1), np.sum(ada.estimator_weights_)
    
    # 
    T = ada.n_estimators
    U = np.zeros((n, T))
    w = np.array(ada.estimator_weights_)
    
    # Add a column to U for each of all T hypothesis 
    for t in range(T):
        h = ada.estimators_[t]
        pred = h.predict(X) * 2 - 1 # change from {0, 1} to {-1, 1} # ERROR WAS HERE. 
        U[:,t] = pred * y           # * is entry wise multiplication.

    margins1 = U @ w
    
    # check computations are correct.
    H = ada.decision_function(X)
    margins2 = H * y
    
    assert np.allclose(margins1, margins2)
    
    np.savetxt("export/U" + str(T) + ".txt", U, delimiter=" ", fmt="%.1f") # only {-1, +1} so we don't need precision
    np.savetxt("export/w" + str(T) + ".txt", w, delimiter=" ", fmt="%.10f") # weights, need all the precision we can get
    
    return U, w, np.min(margins1)



