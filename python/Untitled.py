import sklearn
from sklearn.linear_model import Ridge
import numpy as np
from scipy.sparse import csr_matrix

np.random.seed(2018)
print(sklearn.__version__)

y = np.array([175.0, 1.0, -80.0, 200.0, -100.0])
# w = np.array([8., 1., 20., 5., 1.])
w = np.array([2., 2., 3., 2., 3.])
w /= w.sum()
y_w = np.sqrt(w) * y


m = 5
n = 20
X = np.zeros((m, n))
for i in range(m):
    X[i, np.random.choice(n, 10)] = 1.
X_sp = csr_matrix(X)
X_w = np.sqrt(w.reshape(m, -1)) * X
X_w_sp = csr_matrix(X_w)
X

model = Ridge(alpha=0., fit_intercept=True, copy_X=True)
model.fit(X_w_sp, y_w, sample_weight=None)
model.intercept_, model.coef_

model = Ridge(alpha=0., fit_intercept=True, copy_X=True)
model.fit(X_sp, y, sample_weight=w)
model.intercept_, model.coef_

c2 = .1
w2 = w * c2

model = Ridge(alpha=0., fit_intercept=True, copy_X=True)
model.fit(X_sp, y, sample_weight=w)
model.coef_ / (c * c2)

X = np.zeros((m, n)) 
for i in range(m):
    X[i, np.random.choice(n, 10)] = 1.
X = np.hstack([np.ones((m, 1)), X])
X_sp = csr_matrix(X)

model = Ridge(alpha=0., fit_intercept=False, copy_X=False)
model.fit(X_sp, y, sample_weight=w)

