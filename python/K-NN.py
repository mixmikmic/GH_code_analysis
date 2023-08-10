import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# First we need to create some data. For that, we'll be using a proability distribution
# We'll try to define some parameters for the distribution
mean_01 = np.asarray([0., 2.])
sigma_01 = np.asarray([[1.0, 0.0], [0.0, 1.0]])

mean_02 = np.asarray([4., 0.])
sigma_02 = np.asarray([[1.0, 0.0], [0.0, 1.0]])


data_01 = np.random.multivariate_normal(mean_01, sigma_01, 500)
data_02 = np.random.multivariate_normal(mean_02, sigma_02, 500)
print data_01.shape, data_02.shape

plt.figure(0)
#plt.xlim(-2, 10)
#plt.ylim(-2, 10)
plt.grid('on')
plt.scatter(data_01[:, 0], data_01[:, 1], color='red')
plt.scatter(data_02[:, 0], data_02[:, 1], color='green')
plt.show()

labels = np.zeros((1000, 1))
labels[500:, :] = 1.0

data = np.concatenate([data_01, data_02], axis=0)
print data.shape

ind = range(1000)
np.random.shuffle(ind)

print ind[:10]

data = data[ind]
labels = labels[ind]

print data.shape, labels.shape

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        # compute distance from each point and store in dist
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

x_test = np.asarray([4.0, -2.0])
knn(x_test, data, labels)

# split the data into training and testing
split = int(data.shape[0] * 0.75)

X_train = data[:split]
X_test = data[split:]

y_train = labels[:split]
y_test = labels[split:]

print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

# create a placeholder for storing test predictions
preds = []

# run a loop over every testing example and store the predictions
for tx in range(X_test.shape[0]):
    preds.append(knn(X_test[tx], X_train, y_train))
preds = np.asarray(preds).reshape((250, 1))
print preds.shape

print 100*(preds == y_test).sum() / float(preds.shape[0])

