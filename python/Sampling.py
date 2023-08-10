get_ipython().magic('pylab inline')
import numpy
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from sampler import DTWSampler

# preparing data, nothing very interesting here...

data = []
data.append(numpy.loadtxt("data/Xi_ref.txt"))
data.append(numpy.loadtxt("data/Xi_0.txt"))
data.append(numpy.loadtxt("data/Xi_1.txt"))

d = data[0].shape[1]

max_sz = max([ts.shape[0] for ts in data])
npy_arr = numpy.zeros((len(data), max_sz, d)) + numpy.nan
for idx, ts in enumerate(data):
    sz = ts.shape[0]
    npy_arr[idx, :sz] = ts

print(npy_arr.shape)

print(npy_arr[1, :, 0])

arr = numpy.zeros((10, 15, 2))
arr[:, :, 0] = dJ
arr[:, :, 1] = bleu


npy_arr = npy_arr.reshape(-1, max_sz * d)
print(npy_arr.shape)

s = DTWSampler(scaling_col_idx=0, reference_idx=0, d=d, interp_kind="linear")
transformed_array = s.fit_transform(npy_arr)

print(transformed_array.shape)

pylab.figure(figsize=(15,5))
for i in range(d):
    pylab.subplot(2, 3, i + 1)  # Original data
    if i == 0:
        pylab.title("Reference dimension")
    else:
        pylab.title("Dimension %d" % i)
    if i == 0:
        pylab.ylabel("Original data")
    for ts in npy_arr.reshape((len(data), -1, d)):
        sz = ts.shape[0]
        pylab.plot(numpy.arange(sz), ts[:, i])

    pylab.subplot(2, 3, i + d + 1)  # Transformed data
    if i == 0:
        pylab.ylabel("Resampled data")
    for ts in transformed_array.reshape((len(data), -1, d)):
        sz = ts.shape[0]
        pylab.plot(numpy.arange(sz), ts[:, i])
pylab.show()

