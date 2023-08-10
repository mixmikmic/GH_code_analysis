import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# we will use the following to plot the cost surface
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

# Generate some data
num_points = 1000
x_data,y_data = [],[]

# Define theta parameters for producing test data - Gradient Descent should produce similar values
t0,t1 = 0.4,0.2

for i in xrange(num_points):
    x = np.random.normal(0.0,0.55)
    y = t0 + (x * t1) + np.random.normal(0.0,0.03)
    x_data.append(x)
    y_data.append(y)

# Plot the test data
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
plt.plot(x_data,y_data, 'rx')
plt.title('Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# size of dataset
m = np.shape(x_data)[0]
# Generate the feature matrix
X = np.column_stack((np.ones(m),x_data)) 
# Initialise theta vector
theta = np.array(np.zeros(2)) 

def Cost(X,theta,m,y):

    h = np.dot(X,theta)
    S = np.sum((h - np.transpose(y))**2)
    J = S / (m) # or 2*m

    return J

# initial cost
cost = Cost(X,theta,m,y_data)
print "initial cost: ", cost

def GradientDescent(X,y,theta,alpha,iterations,m):
    xTrans = X.transpose() 
    for i in xrange(iterations):

        h = np.dot(X,theta)
        errors = h - y 
        theta_change = (alpha/m) * np.dot(xTrans,errors)
        theta = theta - theta_change 

    return theta

# -- Define hyperparameters and run Gradient Descent -- #
# learning rate
alpha = 0.01 
# No. iterations for Gradient Descent
iterations = 1500
# Run Gradient Descent
theta = GradientDescent(X,y_data,theta,alpha,iterations,m)

# new cost after optimisation
cost = Cost(X,theta,m,y_data)
print "theta: ", theta, " cost: ", cost

# plot the hypothesis with the learnt fitting values
fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
h = np.dot(X,theta) 
plt.plot(x_data,y_data, 'rx')
plt.plot(x_data,h, label="linear regression")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()

# generate theta values and produce a 2D meshgrid
theta0_vals = np.linspace(theta[0]-5,theta[0]+5,200)
theta1_vals = np.linspace(theta[1]-2,theta[1]+2,200)
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

# initialize J_vals to a matrix of 0's
J_vals = np.array(np.zeros((len(theta1_vals),len(theta0_vals))))

# Calculate the cost for each theta0 theta1 combination
for j in xrange(len(theta1_vals)):
    for i in xrange(len(theta0_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])   
        J_vals[j,i] = Cost(X,t,m,y_data)

fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T0, T1, J_vals, cmap=cm.coolwarm)
plt.xlabel('$\Theta_0$'); plt.ylabel('$\Theta_1$')
ax.set_zlabel('J($\Theta$)')
plt.title('Cost Surface')
plt.subplots_adjust(left=0.001,right=0.99)
plt.show()

fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
CS = plt.contour(theta0_vals, theta1_vals, J_vals,np.logspace(-1, 1, 20))
plt.scatter(theta[0],theta[1],marker='x',color='r',s=50)
plt.clabel(CS, inline=1, fontsize=10)	
plt.xlim(theta[0]-4,theta[0]+4)
plt.ylim(theta[1]-2,theta[1]+2)
plt.xlabel('$\Theta_0$'); plt.ylabel('$\Theta_1$')
plt.title('Cost Surface J($\Theta$)')
plt.show()

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

theta,Jhist = GradientDescent_hist(X,y_data,theta,alpha,iterations,m)

fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
plt.plot(xrange(Jhist.size), Jhist, "-b", linewidth=2 )
plt.title("Convergence of Cost Function")
plt.xlabel('Number of iterations')
plt.ylabel('J($\Theta$)')
plt.show()

def NormEq(X,y):
    return np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

# Use Normal Equation
theta_normal = NormEq(X,y_data)

# new cost and theta after applying the Normal Equation
cost = Cost(X,theta_normal,m,y_data)
print "theta: ", theta_normal, " cost: ", cost

fig = plt.figure(figsize=(7, 5),facecolor='w', edgecolor='k')
h_gradient = np.dot(X,theta) 
h_normal = np.dot(X,theta_normal) 
plt.plot(x_data,y_data, 'rx')
plt.plot(x_data,h_gradient, label="Gradient Descent", color='b',linewidth=2.0)
plt.plot(x_data,h_normal, label="Normal Eq", color='g',linewidth=1.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

