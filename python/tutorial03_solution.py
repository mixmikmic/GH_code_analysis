# Import libraries

# math library
import numpy as np

# visualization library
get_ipython().magic('matplotlib inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x','pdf')
import matplotlib.pyplot as plt

# machine learning library
from sklearn.linear_model import LinearRegression

# 3d visualization
from mpl_toolkits.mplot3d import axes3d

# computational time
import time

# import data with numpy
data = np.loadtxt('data/profit_population.txt', delimiter=',')

# number of training data
n = data.shape[0] #YOUR CODE HERE
print('Number of training data=',n)

# print
print(data[:10,:])
print(data.shape)
print(data.dtype)

x_train = data[:,0]
y_train = data[:,1]

plt.figure(1)
plt.scatter(x_train, y_train, s=30, c='r', marker='x', linewidths=1) #YOUR CODE HERE
plt.title('Training data')
plt.xlabel('Population size (x 10$)')
plt.ylabel('Profit (x 10k$)')
plt.show()

# construct data matrix
X = np.ones([n,2]) 
X[:,1] = x_train
print(X.shape)
print(X[:5,:])


# parameters vector
w = np.array([0.2,-1.4])[:,None] # [:,None] adds a singleton dimension
print(w.shape)
#print(w)


# predictive function definition
def f_pred(X,w): 
    f = X.dot(w) #YOUR CODE HERE
    return f 


# Test predicitive function 
y_pred = f_pred(X,w)
print(y_pred[:5])

# loss function definition
def loss_mse(y_pred,y): 
    n = len(y)
    loss = 1/n* (y_pred - y).T.dot(y_pred - y) #YOUR CODE HERE
    return loss


# Test loss function 
y = y_train[:,None] # label 
#print(y.shape)
y_pred = f_pred(X,w) # prediction
loss = loss_mse(y_pred,y)
print(loss)

# gradient function definition
def grad_loss(y_pred,y,X):
    n = len(y)
    grad = 2/n* X.T.dot(y_pred-y) #YOUR CODE HERE
    return grad


# Test grad function 
y_pred = f_pred(X,w)
grad = grad_loss(y_pred,y,X)
print(grad)    

# gradient descent function definition
def grad_desc(X, y , w_init=np.array([0,0])[:,None] ,tau=0.01, max_iter=500):

    L_iters = np.zeros([max_iter]) # record the loss values
    w_iters = np.zeros([max_iter,2]) # record the loss values
    w = w_init # initialization
    for i in range(max_iter): # loop over the iterations
        y_pred = f_pred(X,w) # linear predicition function #YOUR CODE HERE
        grad_f = grad_loss(y_pred,y,X) # gradient of the loss #YOUR CODE HERE
        w = w - tau* grad_f # update rule of gradient descent #YOUR CODE HERE
        L_iters[i] = loss_mse(y_pred,y) # save the current loss value 
        w_iters[i,:] = w[0],w[1] # save the current w value 
        
    return w, L_iters, w_iters


# run gradient descent algorithm 
start = time.time()
w_init = np.array([0.2,-1.4])[:,None]
tau = 0.01
max_iter = 20
w, L_iters, w_iters = grad_desc(X,y,w_init,tau,max_iter)
print('Time=',time.time() - start)
print(L_iters[-1])
print(w)


# plot
plt.figure(2)
plt.plot(np.array(range(max_iter)), L_iters)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# linear regression model
x_pred = np.linspace(0,25,100) #YOUR CODE HERE
y_pred = w[0] + w[1]* x_pred #YOUR CODE HERE

# plot
plt.figure(3)
plt.scatter(x_train, y_train, s=30, c='r', marker='x', linewidths=1)
plt.plot(x_pred, y_pred,label='gradient descent optimization'.format(i=1))
plt.legend(loc='best')
plt.title('Training data')
plt.xlabel('Population size (x 10k)')
plt.ylabel('Profit $(x 10k)')
plt.show()

# run linear regression with scikit-learn
start = time.time()
lin_reg_sklearn = LinearRegression()
lin_reg_sklearn.fit(x_train[:,None], y_train) # learn the model parameters #YOUR CODE HERE
print('Time=',time.time() - start)


# compute loss value
w_sklearn = np.zeros([2,1])
w_sklearn[0,0] = lin_reg_sklearn.intercept_
w_sklearn[1,0] = lin_reg_sklearn.coef_
print(w_sklearn)
loss_sklearn = loss_mse(f_pred(X,w_sklearn),y_train[:,None])
print('loss sklearn=',loss_sklearn)
print('loss gradient descent=',L_iters[-1]) 


# plot
y_pred_sklearn = w_sklearn[0] + w_sklearn[1]* x_pred
plt.figure(3)
plt.scatter(x_train, y_train, s=30, c='r', marker='x', linewidths=1)
plt.plot(x_pred, y_pred,label='gradient descent optimization'.format(i=1))
plt.plot(x_pred, y_pred_sklearn,label='Scikit-learn optimization'.format(i=2))
plt.legend(loc='best')
plt.title('Training data')
plt.xlabel('Population size (x 10k)')
plt.ylabel('Profit $(x 10k)')
plt.show()

# run gradient descent algorithm 
start = time.time()
w_init = np.array([0.2,-1.4])[:,None]
tau = 0.01
max_iter = 1000
w, L_iters, w_iters = grad_desc(X,y,w_init,tau,max_iter)
print('Time=',time.time() - start)
print(L_iters[-1])
print(w)


# plot
y_pred = w[0] + w[1]* x_pred
plt.figure(4)
plt.scatter(x_train, y_train, s=30, c='r', marker='x', linewidths=1)
plt.plot(x_pred, y_pred,label='gradient descent optimization'.format(i=1))
plt.plot(x_pred, y_pred_sklearn,label='Scikit-learn optimization'.format(i=2))
plt.legend(loc='best')
plt.title('Training data')
plt.xlabel('Population size (x 10k)')
plt.ylabel('Profit $(x 10k)')
plt.show()

# Predict profit for a city with population of 45000
print('Profit would be',w.T.dot([1,4.5])[0]*10000) #YOUR CODE HERE

# plot gradient descent 
def plot_gradient_descent(X,y,w_init,tau,max_iter):
    
    def f_pred(X,w):
        f = X.dot(w) 
        return f
    
    def loss_mse(y_pred,y):
        n = len(y)
        loss = 1/n* (y_pred - y).T.dot(y_pred - y)
        return loss
    
    def grad_desc(X, y , w_init=np.array([0,0])[:,None] ,tau=0.01, max_iter=500):

        L_iters = np.zeros([max_iter]) # record the loss values
        w_iters = np.zeros([max_iter,2]) # record the loss values
        w = w_init # initialization
        for i in range(max_iter): # loop over the iterations
            y_pred = f_pred(X,w) # linear predicition function
            grad_f = grad_loss(y_pred,y,X) # gradient of the loss
            w = w - tau* grad_f # update rule of gradient descent
            L_iters[i] = loss_mse(y_pred,y) # save the current loss value
            w_iters[i,:] = w[0],w[1] # save the current w value

        return w, L_iters, w_iters

    # run gradient descent
    w, L_iters, w_iters = grad_desc(X,y,w_init,tau,max_iter)
    
    # Create grid coordinates for plotting a range of L(w0,w1)-values
    B0 = np.linspace(-10, 10, 50)
    B1 = np.linspace(-1, 4, 50)
    xx, yy = np.meshgrid(B0, B1, indexing='xy')
    Z = np.zeros((B0.size,B1.size))  

    # Calculate loss values based on L(w0,w1)-values
    for (i,j),v in np.ndenumerate(Z):
        Z[i,j] = loss_mse(f_pred(X,w=[[xx[i,j]],[yy[i,j]]]),y)

    # 3D visualization
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # Left plot
    CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
    ax1.scatter(w[0],w[1], c='r')
    ax1.plot(w_iters[:,0],w_iters[:,1])

    # Right plot
    ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax2.set_zlabel('Loss $L(w_0,w_1)$')
    ax2.set_zlim(Z.min(),Z.max())
    #ax2.view_init(elev=10, azim=-120)

    # plot gradient descent
    Z2 = np.zeros([max_iter])
    for i in range(max_iter):
        w0 = w_iters[i,0]
        w1 = w_iters[i,1]
        Z2[i] = loss_mse(f_pred(X,w=[[w0],[w1]]),y)
    ax2.plot(w_iters[:,0],w_iters[:,1],Z2)
    ax2.scatter(w[0],w[1],loss_mse(f_pred(X,w=[w[0],w[1]]),y), c='r')

    # settings common to both plots
    for ax in fig.axes:
        ax.set_xlabel(r'$w_0$', fontsize=17)
        ax.set_ylabel(r'$w_1$', fontsize=17)
    

# run plot_gradient_descent function
w_init = np.array([0.2,-1.4])[:,None]
tau = 0.01
max_iter = 200
plot_gradient_descent(X,y,w_init,tau,max_iter) 

# run plot_gradient_descent function
w_init = np.array([0.2,-1.4])[:,None]
tau = 0.01
max_iter = 2000
plot_gradient_descent(X,y,w_init,tau,max_iter) 







