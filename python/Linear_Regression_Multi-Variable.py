import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data = np.loadtxt("../data/ex1data2.txt",delimiter=',')
x_data = data[:,[0,1]]
y_data = data[:,2]

# #  --- Plot test data --- #
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
ax.set_zlabel('y')
plt.title('Input Data')
plt.subplots_adjust(left=0.001,right=0.99)
ax.view_init(35, -30)
plt.show()

# size of dataset
m = len(y_data)
# n features
n = np.shape(x_data)[1] 
print "Number of data points:", m
print "Number of features in data:", n

def FeatureNorm(X):

    Xnorm = np.zeros(np.shape(X))

    mu    = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in xrange(X.shape[1]):
        mu[:,i]    = np.mean(X[:,i])
        sigma[:,i] = np.std(X[:,i])
        Xnorm[:,i] = (X[:,i] - float(mu[:,i]))/float(sigma[:,i])
    return Xnorm,mu,sigma

# Feature Scaling - normalise the features to ensure equal weighting
Xnorm,mu,sigma = FeatureNorm(x_data)
# Generate the feature matrix - add column of ones
X = np.column_stack((np.ones((m,1)), Xnorm))
# Initialise theta vector
theta = np.array(np.zeros(n+1)) 

def Cost(X,theta,m,y):

    h = np.dot(X,theta)
    S = np.sum((h - np.transpose(y))**2)
    J = S / (m) # or 2*m

    return J

# initial cost
cost = Cost(X,theta,m,y_data)
print "initial cost: ", cost

def GradientDescent_hist(X,y,theta,alpha,iterations,m):

    Jhist = np.zeros((iterations,1))
    xTrans = X.transpose() 
    for i in xrange(iterations):
        h = np.dot(X,theta)
        errors = h - np.transpose(y)  
        theta_change = (alpha/m) * np.dot(xTrans,errors)
        theta = theta - theta_change 

        Jhist[i] = Cost(X,theta,m,y)

    return theta,Jhist

# -- Define hyperparameters and run Gradient Descent -- #
# learning rate
alpha = 0.01 
# No. iterations for Gradient Descent
iterations = 1000
# Run Gradient Descent
theta,Jhist = GradientDescent_hist(X,y_data,theta,alpha,iterations,m)

# Plot covergence of cost
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
plt.plot(xrange(Jhist.size), Jhist, "-b", linewidth=2 )
plt.title("Convergence of Cost Function")
plt.xlabel('Number of iterations')
plt.ylabel('J($\Theta$)')
plt.show()

# Calculate the cost after fitting
cost = Cost(X,theta,m,y_data)
print "theta: ", theta, "\ncost: ", cost

# plot the hypothesis with the learnt fitting values
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
x1 = np.linspace(np.min(x_data[:,0]), np.max(x_data[:,0]), 500)
x2 = np.linspace(np.min(x_data[:,1]), np.max(x_data[:,1]), 500)
x1norm = np.linspace(np.min(Xnorm[:,0]), np.max(Xnorm[:,0]), 500)
x2norm = np.linspace(np.min(Xnorm[:,1]), np.max(Xnorm[:,1]), 500)
h = np.dot(np.column_stack((np.ones(500),np.column_stack((x1norm,x2norm)) )),theta) 
ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
ax.plot(x1,x2,h,c='r',linewidth=1.5)
plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
ax.set_zlabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.subplots_adjust(left=0.001,right=0.99)
ax.view_init(35, -30)

def NormEq(X,y):
    return np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

# Generate the feature matrix - add column of ones
X = np.column_stack((np.ones((m,1)), x_data))

# Use Normal Equation
theta_normal = NormEq(X,y_data)

# Calculate the cost after fitting
cost = Cost(X,theta_normal,m,y_data)
print "theta: ", theta_normal, "\ncost: ", cost

# plot the hypothesis with the learnt fitting values
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
x1 = np.linspace(np.min(x_data[:,0]), np.max(x_data[:,0]), 500)
x2 = np.linspace(np.min(x_data[:,1]), np.max(x_data[:,1]), 500)
x1norm = np.linspace(np.min(Xnorm[:,0]), np.max(Xnorm[:,0]), 500)
x2norm = np.linspace(np.min(Xnorm[:,1]), np.max(Xnorm[:,1]), 500)
h_gradient = np.dot(np.column_stack((np.ones(500),np.column_stack((x1norm,x2norm)) )),theta) 
h_normal   = np.dot(np.column_stack((np.ones(500),np.column_stack((x1,x2)) )),theta_normal) 
ax.scatter(x_data[:,0], x_data[:,1],y_data, cmap=cm.coolwarm)
ax.plot(x1,x2,h_gradient,c='r',linewidth=2.5,label="gradient descent")
ax.plot(x1,x2,h_normal,c='g',linewidth=1.5,label="normal Eq")
plt.xlabel('$x_0$'); plt.ylabel('$x_1$')
ax.set_zlabel('y')
plt.title('Linear Regression')
plt.legend()
plt.subplots_adjust(left=0.001,right=0.99)
ax.view_init(35, -30)
plt.show()

