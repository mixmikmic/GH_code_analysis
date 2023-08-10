import numpy as np

# Generate 21 random integers and sort them
X = np.random.randint(100, size=21)
X.sort()
print X

print '50th percentile:', np.percentile(X, 50)
print '95th percentile:', np.percentile(X, 95)
print '11th percentile:', np.percentile(X, 11)

