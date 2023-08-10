# Set the directory where the sentiment data will be deployed. Use a place local to your computer.
datadir = '/almacen/Media/Meta/ml/sentiment/'

import os.path
sentiment_dir = os.path.join(datadir,'txt_sentoken')

if not os.path.isdir(sentiment_dir):
    import urllib2
    import tarfile
    src = urllib2.urlopen( 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz' )
    with tarfile.open( fileobj=src, mode='r|gz' ) as sdata:
        sdata.extractall( path=datadir )
    src.close

from sklearn.datasets import load_files
data = load_files( sentiment_dir, shuffle=True )

idx = 2
print "Value =", data.target[idx], "\n-------\n", data.data[idx]

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the object
vect = TfidfVectorizer( min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True )

# Do the indexing and obtain the term-document matrix
X = vect.fit_transform(data.data)

X.shape

# See the 8 words with lowest IDF score
reversed_dict = { v:k for k,v in vect.vocabulary_.iteritems() }
lowest = np.argpartition(vect.idf_,8)[:8]
key = lambda x : vect.idf_[x]
for idf in sorted( lowest, key=key ):
    print "{:10} {}".format( reversed_dict[idf], vect.idf_[idf] )

# See the 8 words with highest IDF score (the "rare" words)
for i in np.argpartition(vect.idf_,-8)[-8:]:
    print "{:10} {}".format( reversed_dict[i], vect.idf_[i] )

# How many terms were discarded in the indexing?
len( vect.stop_words_)

# do we have sparse data (as we should in a typical TF/IDF array)?
import scipy.sparse as sp
sp.issparse(X)

# The usual train/test split
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state

rs = check_random_state( 1352 )
Xr, Xt, yr, yt = train_test_split( X, data.target, test_size=0.30, random_state=rs )

# Let's try a random forest. We train with an ensemble of 30 trees
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20).fit( Xr, yr )
yp = clf.predict( Xt )

# How good did we get?
from sklearn.metrics import accuracy_score, confusion_matrix

print "Accuracy =", accuracy_score( yt, yp )

# See the confusion matrix obtained
print confusion_matrix( yt, yp )

from sklearn.grid_search import GridSearchCV

# The parameters we are going to try: number of trees
param_grid = [
  {'n_estimators': range(10,200,10) },
 ]

# Do a grid search over the parameter space, performing 5-fold cross-validation at each datapoint
# This will take quite a while
clfs = GridSearchCV( RandomForestClassifier(), param_grid, cv=5, n_jobs=2 )
clfs.fit(Xr, yr)

# See what we got
print "Best parameters set found on development set:\n"
print(clfs.best_params_)

print "\nGrid scores on development set:"
for params, mean_score, scores in clfs.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

# Extract the cross-validation scores
trees = [ p.parameters['n_estimators'] for p in clfs.grid_scores_ ]
result = np.vstack( [p.cv_validation_scores for p in clfs.grid_scores_] )
result_mean = np.mean( result, axis=1 )
result_std = np.std( result, axis=1 )

# And plot them
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure( 1, (7*1.62,7), dpi=200 )
from operator import itemgetter
get_means = itemgetter('mean')
plt.plot( trees, result_mean );
plt.fill_between( trees, result_mean-result_std, result_mean+result_std, alpha=0.15 );
plt.xlabel( 'number of trees');
plt.ylabel( 'accuracy' );

clf_best = RandomForestClassifier(n_estimators=150).fit( Xr, yr )
yp = clf_best.predict( Xt )

print "Best accuracy =", accuracy_score( yt, yp )

