import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')

mean1 = np.array([0,0])
mean2 = np.array([2,0])

P = np.array([[1,1], [-1,1]]) # matrice de changement de base, pour pencher ma distribution
cov1 = np.dot(np.dot(P, np.array([[0.1,0],[0,1]])), np.linalg.inv(P))
cov2 = np.dot(np.dot(P, np.array([[1,0],[0,0.1]])), np.linalg.inv(P))

x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

plt.axis([-3,5,-3,3])
plt.scatter(x, y, color="blue")

one = np.ones(len(x)).reshape(len(x), 1) # for the bias

X, y = np.concatenate([one, x.reshape(len(x), 1)], axis=1) , y.reshape(len(y), 1)

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, d=2):
    return (np.dot(x1, x2) + 1)**d

def gaussian_kernel(x1, x2, gamma=1):
    # experimentally : the bigger gamma is, the higher is the capacity of the model
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def exponential_kernel(x1, x2, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2))

def tanh_kernel(x1, x2, alpha=1):
    # experimentally : the bigger alpha is, the higher is the capacity of the model
    return np.tanh(alpha * np.dot(x1, x2))

def rational_quadratic_kernel(x1, x2, c=1):
    return 1 - (np.linalg.norm(x1 - x2)**2)/(c + np.linalg.norm(x1 - x2)**2)

def power_kernel(x1, x2, d=4):
    return - np.linalg.norm(x1 - x2)**d

def K(X1, X2, kernel=polynomial_kernel):
    return np.array([[kernel(X1[i], X2[j]) for i in range(X1.shape[0])] for j in range(X1.shape[0])])

def kappa(X1, x2, kernel=polynomial_kernel):
    return np.array([kernel(X1[i], x2) for i in range(len(X1))])

S = K(X, X)
lamda = 0.1
print y.shape
print S.shape
print np.linalg.inv(S + lamda * np.eye(len(S)))

def f(x, S, kernel):
    return np.dot(np.dot(y.transpose(), np.linalg.inv(S + lamda * np.eye(len(S)))), kappa(X, x, kernel=kernel))

def F(x_vect, kernel):
    S = K(X, X, kernel=kernel)
    return np.array([f(x_vect[i], S, kernel) for i in range(len(x_vect))])

plt.rcParams["figure.figsize"] = (12,12)

new_x = np.concatenate([np.ones(101).reshape(101,1), np.linspace(-4, 5, 101).reshape(101, 1)], axis=1)

y_linear = F(new_x, linear_kernel)
y_poly = F(new_x, polynomial_kernel)
y_gauss = F(new_x, gaussian_kernel)
y_tanh = F(new_x, tanh_kernel)
y_rq = F(new_x, rational_quadratic_kernel)
y_power = F(new_x, power_kernel)

plt.subplot(331)
plt.title("Linear")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_linear)

plt.subplot(332)
plt.title("Polynomial 2")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_poly)

plt.subplot(333)
plt.title("Gaussian")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_gauss)

plt.subplot(334)
plt.title("Tanh")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_tanh)

plt.subplot(335)
plt.title("Rational quadratic")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_rq)

plt.subplot(336)
plt.title("Power 4")
plt.axis([-3,5,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_power)





