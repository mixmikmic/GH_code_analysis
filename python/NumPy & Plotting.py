import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

d = {}
n = 10000
for i in range(n):
    x = np.random.randint(0, 10)
    d[x] = d.get(x, 0) + 1

plt.bar(list(d.keys()), d.values(), color='g')
plt.show()

n = 1000
gaussian_numbers = np.random.randn(n)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Sampling
beta = 2
x = []
y = []

min_x = 0
max_x = 10

for i in range(100):
    x_val = np.random.uniform(min_x, max_x)
    x.append(x_val)
    epsilon = np.random.normal()
    
    y_val = beta*x_val + epsilon
    y.append(y_val)

#print("Feature/Independent Variable:", x)
#print("Dependent Variable:", y)

cov = np.cov(x, y)
print("\nCovariance Matrix")
print(cov)

print("\nSlope Estimate (beta_hat):", cov[0][1] / cov[0][0])

plt.plot(x, y, 'ro')
plt.title("Scatter Plot: y = beta*x")
plt.xlabel("Control Variable")
plt.ylabel("Observed Variable")
plt.show()

# Sampling
beta = [10, 2, 3]
x = []
y = []

min_x = 0
max_x = 100

for i in range(100):
    x_val = np.random.uniform(min_x, max_x)
    x.append(x_val)
    epsilon = np.random.normal()
    
    # Use the NumPy dot product
    y_val = np.dot(beta, [1, x_val, x_val**2]) + epsilon
    y.append(y_val)

#print("Feature/Independent Variable:", x)
#print("Dependent Variable:", y)

cov = np.cov(x, y)
print("\nCovariance Matrix")
print(cov)

plt.plot(x, y, 'ro')
plt.title("Scatter Plot: y = beta0 + beta1*x + beta2*x^2")
plt.xlabel("Control Variable")
plt.ylabel("Observed Variable")
plt.show()



