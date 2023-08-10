# Import NumPy and seed random number generator to make generated matrices deterministic
import numpy as np
np.random.seed(2)

# Create a matrix with random entries
A = np.random.rand(4, 4)
print(A)

# Compute eigenvectors of A
evalues, evectors = np.linalg.eig(A)

print("Eigenvalues: {}".format(evalues))
print("Eigenvectors: {}".format(evectors))

Lambda = np.linalg.inv(evectors).dot(A.dot(evectors))
print(Lambda)

# Create a symmetric matrix
S = A + A.T

# Compute eigenvectors of S and print eigenvalues
lmbda, U = np.linalg.eig(S)
print(lmbda)

# R matrix
R = U.T

# Diagonalise S
Lambda = R.dot(S.dot(R.T))
print(Lambda)

# Create starting vector
x0 = np.random.rand(S.shape[0])

# Perform power iteration
for i in range(10):
    x0 = S.dot(x0)
    x0 = x0/np.linalg.norm(x0)
x1 = S.dot(x0)

# Get maxiumum exact eigenvalue (absolute value)
eval_max_index = abs(lmbda).argmax()
max_eig = lmbda[eval_max_index]

# Print estimated max eigenvalue and error 
max_eig_est = np.sign(x1.dot(x0))*np.linalg.norm(x1)/np.linalg.norm(x0)
print("Estimate of largest eigenvalue: {}".format(max_eig_est))
print("Error: {}".format(abs(max_eig - max_eig_est)))

# Create starting vector
x0 = np.random.rand(S.shape[0])

# Get eigenvector associated with maxiumum eigenvalue
eval_max_index = abs(lmbda).argmax()
evec_max = U[:,eval_max_index]

# Make starting vector orthogonal to eigenvector associated with maximum 
x0 = x0 - x0.dot(evec_max)*evec_max

# Perform power iteration
for i in range(10):
    x0 = S.dot(x0)
    x0 = x0/np.linalg.norm(x0)
x1 = S.dot(x0)

# Print estimated max eigenvalue and error
max_eig_est = np.sign(x1.dot(x0))*np.linalg.norm(x1)/np.linalg.norm(x0)
print("Estimate of largest eigenvalue: {}".format(max_eig_est))
print("Error: {}".format(abs(max_eig - max_eig_est)))   

# Get second largest eigenvalue
print("Second largest eigenvalue (exact): {}".format(lmbda[np.argsort(abs(lmbda))[-2]]))  

rayleigh_quotient = x1.dot(S).dot(x1)/(x1.dot(x1))
print("Rayleigh_quotient: {}".format(rayleigh_quotient))

A = np.array([[0, 2, 0.9663], [0.545, 0 ,0], [0, 0.8, 0]])

# Create starting vector
x0 = np.random.rand(A.shape[0])

# Perform power iteration
for i in range(10):
    x0 = A.dot(x0)
    x0 = x0/np.linalg.norm(x0)

# Normalise eigenvector using l1 norm
x0 = x0/np.linalg.norm(x0, 1)

# Print estimated eigenvector associated with largest eigenvalue
print("Estimate of eigenvector for the largest eigenvalue: {}".format(x0))

# Print estimated max eigenvalue (using Rayleigh quotient)
print("Estimate of largest eigenvalue: {}".format(x0.dot(A).dot(x0)/x0.dot(x0)))

