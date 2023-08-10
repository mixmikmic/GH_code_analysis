import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def graddesc(A, b, x, tol):
    # Compute the negative gradient r = A^T(b-Ax)
    r = np.dot(A.transpose(),b-np.dot(A,x))
    # Start with an empty array
    xout = [x]
    while la.norm(r,2) > tol:
        # If the gradient is bigger than the tolerance
        Ar = np.dot(A,r)
        alpha = np.dot(r,r)/np.dot(Ar,Ar)
        x = x + alpha*r
        xout.append(x)
        r = r-alpha*np.dot(A.transpose(),Ar)
    return np.array(xout).transpose()

A = np.array([[1,2], [2,1], [-1,0]])
b = np.array([10, -1, 0])
tol = 1e-4
x = np.zeros(2)

traj = graddesc(A, b, x, tol)

get_ipython().magic('matplotlib inline')

# First create a contour plot to see where the minimum would be

# Define the function we aim to minimize
def f(x):
    return np.dot(np.dot(A,x)-b,np.dot(A,x)-b)

# Create a mesh grid 
xx = np.linspace(-3,1,100)
yy = np.linspace(0,6,100)
X, Y = np.meshgrid(xx, yy)
Z = np.zeros(X.shape)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

# Get a nice monotone colormap
cmap = plt.cm.get_cmap("coolwarm")

# Plot the contours and the trajectory
plt.contourf(X, Y, Z, cmap = cmap)
plt.plot(traj[0,:], traj[1,:], 'o-k')
plt.show()

def rosenbrock(X, Y):
    return ((1-X)**2)+100*(Y-X**2)**2

xx = np.linspace(-2,2,100)
yy = np.linspace(-2,2,100)
X, Y = np.meshgrid(xx,yy)
Z = rosenbrock(X, Y)

get_ipython().magic('matplotlib inline')
plt.figure()
levels = [0.1,0.5,1.0,2.0,5.0,10.0,50.0,100.0,200.0,400.0,600.0]
cmap = plt.cm.get_cmap("coolwarm")

cp = plt.contour(X,Y,Z,levels,cmap = cmap)

plt.title('Level sets of Rosenbrock function')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([1],[1],'ro')
plt.show()



