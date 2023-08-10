import mxnet
import mxnet.ndarray as nd
import urllib
import gzip

with gzip.open(urllib.request.urlopen("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz")) as f:
    data = [eval(l) for l in f]

data[0]

users = [d['reviewerID'] for d in data]

items = [d['asin'] for d in data]

ratings = [d['overall'] for d in data]











