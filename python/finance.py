get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def update_network(leaking_rate,time_constant,X,y_fb,u,W,Wfb,Win,f):
    return (1-leaking_rate*time_constant) * X + time_constant * f(Win.dot(u)+W.dot(X) + Wfb.dot(y_fb) + np.random.randn(*y_fb.shape)*0.001)

def compute_output(X,Wout):
    return Wout.T.dot(X)
    #return np.tanh(Wout.T.dot(X))

N = 300
X = np.random.rand(N) - 0.5
W = np.random.rand(N,N) - 0.5
Wbf = (np.random.rand(N,1) - 0.5)*0
Win = (np.random.rand(N,1) - 0.5)
rhoW = np.max(np.abs(np.linalg.eig(W)[0]))
W *= 0.2 / rhoW
leaking_rate = 0.9
time_constant = 0.06

u = pd.read_csv("../data/MackeyGlass_t17.txt")
u = (u-u.mean())*1./u.std()
u = u.values.reshape(len(u.index),1)
_ = plt.plot(range(len(u))[:1000],u[:1000])

training_size = 3000
rest_size = 500
data_size = 9999


X_full = np.ones((training_size,N))
for i in range(rest_size):
    X = update_network(leaking_rate,time_constant,X,u[i],u[i],W,Wbf,Win,np.tanh)

for i in range(training_size):
    X_full[i,:] = X
    X = update_network(leaking_rate,time_constant,X,u[i+rest_size],u[i+rest_size],W,Wbf,Win,np.tanh)
_=plt.plot(pd.DataFrame(X_full))

Wout = np.linalg.pinv(X_full).dot(u[rest_size:training_size+rest_size,:])#np.arctanh(y[:training_size,:]))Wout = np.linalg.pinv(X_full).dot(y[rest_size:training_size+rest_size,:])#np.arctanh(y[:training_size,:]))
#Wout = np.linalg.pinv(X_full).dot(np.arctanh(y[rest_size:training_size+rest_size,:]))#y[:training_size,:]))
_=plt.plot(pd.DataFrame(Wout))

ans = []
test_size = data_size-training_size-rest_size
for i in range(test_size):
    X = update_network(leaking_rate,time_constant,X,u[i+training_size+rest_size-1],u[i+training_size+rest_size-1],W,Wbf,Win,np.tanh)
    ans.append(compute_output(X,Wout))

_ = plt.plot(range(test_size)[:2000],u[training_size+rest_size:][:2000])
_ = plt.plot(range(test_size)[:2000],ans[:2000])

ans = []
for i in range(training_size-rest_size):
    X = update_network(leaking_rate,time_constant,X,u[i+rest_size-1],u[i+rest_size-1],W,Wbf,Win,np.tanh)
    ans.append(compute_output(X,Wout))

_ = plt.plot(range(training_size-rest_size),u[rest_size:training_size])
_ = plt.plot(range(training_size-rest_size),ans)

ans = []
test_size = 3000
for i in range(test_size):
    if i < 300:
        X = update_network(leaking_rate,time_constant,X,u[i+training_size+rest_size-1],u[i+training_size+rest_size],W,Wbf,Win,np.tanh)
    else:
        X = update_network(leaking_rate,time_constant,X,ans[i-1],u[i+training_size+rest_size],W,Wbf,Win,np.tanh)
    ans.append(compute_output(X,Wout))

_ = plt.plot(range(test_size),u[training_size+rest_size:training_size+rest_size+test_size])
_ = plt.plot(range(test_size),ans)



