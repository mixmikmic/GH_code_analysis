get_ipython().magic('pylab inline')
import pylab
from sklearn.datasets import fetch_mldata
DATA_PATH = '~/data'
mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)

mnist.data.shape
#img = mnist.data[32000,:]

value = 39130
train = mnist.data[:60000]
test = mnist.data[60000:]
img = train[value]
pylab.imshow(img.reshape(28,28), cmap="Greys")
print model.predict(mnist.data[value])

row = mnist.data[0,:] # First row of the array
col = mnist.data[:,0] # First column of the array

print row.shape
print col.shape

print row.sum(), row.max(), row.min()
print col.sum(), col.max(), col.min()

print mnist.data[:10,:] # First ten rows
print mnist.data[:,-10:] # Last ten columns

train = mnist.data[:60000]
test = mnist.data[60000:]
test[0]

#test_sample = None # Fix me
test_sample = test[::100]

test_sample[:,299].mean()

img = mnist.data[0]
print img

pylab.imshow(img.reshape(28, 28), cmap="Greys")

img = mnist.data[35500]
pylab.imshow(img.reshape(28, 28), cmap="Greys")

get_ipython().run_cell_magic('time', '', "from sklearn.neighbors import NearestNeighbors\nmodel = NearestNeighbors(algorithm='brute').fit(train)")

get_ipython().run_cell_magic('time', '', 'query_img = test[0]\n_, result = model.kneighbors(query_img, n_neighbors=4)')

print result

# Display several images in a row
def show(imgs, n=1):
    fig = pylab.figure()
    for i in xrange(0, n):
        fig.add_subplot(1, n, i, xticklabels=[], yticklabels=[])
        if n == 1:
            img = imgs
        else:
            img = imgs[i]
        pylab.imshow(img.reshape(28, 28), cmap="Greys")

show(query_img)
show(train[result[0],:], len(result[0]))



train_labels = mnist.target[:60000]
test_labels = mnist.target[60000:]
test_labels_sample = test_labels[::100]

get_ipython().run_cell_magic('time', '', "from sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier(n_neighbors=4, algorithm='brute').fit(train, train_labels)")

get_ipython().run_cell_magic('time', '', '# Score the model!')

preds = model.predict(test_sample)
errors = [i for i in xrange(0, len(test_sample)) if preds[i] != test_labels_sample[i]]

for i in errors:
    pass # Visualize error image and its nearest neighbors

test_sample = test[::10]
test_labels_sample = test_labels[::10]
preds = model.predict(test_sample)

def plot_cm(cm):
    pylab.matshow(np.log(1+cm))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_sample, preds)

print cm
plot_cm(cm)

# Looks like 2 and 7 have the most confusion

model.predict(mnist.data[11000])





