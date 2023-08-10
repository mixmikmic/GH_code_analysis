import numpy as np
import urllib.request
import tarfile
import os
import zipfile
import gzip
import os
from glob import glob
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n) # will also set self.n = b * bsize

# http://yann.lecun.com/exdb/mnist/

labels_filename = 'train-labels-idx1-ubyte.gz'
images_filename = 'train-images-idx3-ubyte.gz'

url = "http://yann.lecun.com/exdb/mnist/"
with TqdmUpTo() as t:  # all optional kwargs
    urllib.request.urlretrieve(url+images_filename, 'MNIST_'+images_filename, reporthook=t.update_to, data=None)
with TqdmUpTo() as t:  # all optional kwargs
    urllib.request.urlretrieve(url+labels_filename, 'MNIST_'+labels_filename, reporthook=t.update_to, data=None)

# https://www.nist.gov/itl/iad/image-group/emnist-dataset

url = "http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
filename = "gzip.zip"
with TqdmUpTo() as t:  # all optional kwargs
    urllib.request.urlretrieve(url, filename, reporthook=t.update_to, data=None)

zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall('.')
zip_ref.close()

if os.path.isfile(filename):
    os.remove(filename)

# https://github.com/zalandoresearch/fashion-mnist

url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
filename = "train-images-idx3-ubyte.gz"
with TqdmUpTo() as t:  # all optional kwargs
    urllib.request.urlretrieve(url, filename, reporthook=t.update_to, data=None)

url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
filename = "train-labels-idx1-ubyte.gz"
_ = urllib.request.urlretrieve(url, filename)

