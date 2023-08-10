import numpy as np
import matplotlib.pyplot as plt

# Draw points randomly from a normal distribution
X = np.random.randn(150)
print np.sort(X)

# Print list of frequencies of data points
# Second argument is number of bins
# Second output is the locations of bin dividers but we ignore it because there are 301 of them
hist, _ = np.histogram(X, 300)
print 'Data frequencies:', hist

plt.hist(X,300);

# Print frequency of data per bin
hist, bins = np.histogram(X, 15)
print 'Bin dividers:', bins
print 'Binned data frequencies:', hist
plt.hist(X,15);

# Print frequency of data per bin
hist, bins = np.histogram(X, 4)
print 'Bin dividers:', bins
print 'Binned data frequencies:', hist
plt.hist(X,4);

