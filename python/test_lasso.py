import sys
sys.version
sys.version_info

import sys
sys.path.append('../')

import scipy.sparse as sp
import numpy as np

from lasso import SparseLasso

# try the code used in the scikit-learn example. 
skl_ex_X = sp.csc_matrix([[0.,0], [1, 1], [2, 2]])
skl_ex_Y = np.array([0, 1, 2])

sp.csc_matrix(np.array([[0, 1, 2]]))

lasso_toy = SparseLasso(X = skl_ex_X, 
            y = skl_ex_Y,
            lam = 0.1,
            #w = np.array([0., 0.]),
            verbose = True
           )
lasso_toy.run()
lasso_toy.w.toarray()

print(lasso_toy.w.toarray())
print("")
print(lasso_toy.w0)
print("")
print(lasso_toy.objective())
print("")
print(lasso_toy.calc_yhat())

from sklearn import linear_model
lam = 0.1
alpha = lam/(2*3)  # hard coded for 3 sample points. 
clf = linear_model.Lasso(alpha)  # 3 samples  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
clf.fit([[0.,0], [1, 1], [2, 2]], [0., 1, 2])

print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0.,0], [1, 1], [2, 2]]))

# try the code used in the scikit-learn example. 
toy_X = sp.csc_matrix([[0.,1], [1, 2], [2, 3]])
toy_Y = np.array([2., 5, 8])

toy = SparseLasso(X = toy_X, 
            y = toy_Y,
            lam = 0.1,
            #w = np.array([0., 0.]),
            verbose = True
           )

toy.run()

print(toy.w.toarray())
print(toy.w0)

from sklearn import linear_model
lam = 0.1
alpha = lam/(2*3)  # hard coded for 3 sample points. 
clf = linear_model.Lasso(alpha)  # 3 samples  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
clf.fit([[0.,1], [1, 2], [2, 3]], [2., 5, 8])

print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0.,1], [1, 2], [2, 3]]))



