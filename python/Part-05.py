import gzip
from urllib.request import urlretrieve
from tqdm import tqdm
import os
# if you are using the fastAI environment, all of these imports work

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)

# Let's download some data:
data_url = 'http://files.fast.ai/data/aclImdb.tgz'
get_data(data_url, 'data/imdb.tgz')

for pathroute in os.walk('data\\imdb\\aclImdb'):
    next_path = pathroute[1]
    for stop in next_path:
        print(stop)

import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



