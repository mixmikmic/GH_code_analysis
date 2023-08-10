get_ipython().magic('pylab inline')
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('../../data/linear-regression/ex1data1.txt', delimiter=',')
X = data[:, 0].reshape(data[:, 0].shape[0], 1) # Population
Y = data[:, 1].reshape(data[:, 1].shape[0], 1) # profit

# Standardization
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y)

scatter(X, Y)
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
grid()

w = np.array([-0.1941133,  -2.07505268])

def predict(w, X):
    N = len(X)
    yhat = w[1:].dot(X.T) + w[0]
    yhat = yhat.reshape(X.shape)
    return yhat

def momentum_nn(X, Y, w, eta=0.1, gamma=0.5):
    N = len(X)
    v = np.zeros(w.shape[1:])
    v_b = np.zeros(1)
    
    for i in range(N):
        x = X[i]
        y = Y[i]
        yhat = predict(w, x)
        delta = y - yhat
        
        v = gamma*v + 2/N * eta * np.sum(-delta.dot(x))
        v_b = gamma*v_b + 2/N * eta * np.sum(-delta)
        
        w[1:] = w[1:] - v
        w[0] = w[0] - v_b
    return w
    

for i in range(1, 10):
    w = momentum_nn(X, Y, w)
    yhat = predict(w, X)
    
    axes = subplot(3, 3, i)
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
    scatter(X, Y)
    plot(X, yhat, color='red')

