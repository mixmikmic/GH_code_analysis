import sys
print ("Your Python version is", sys.version)

errors = 0

try:
    import numpy as np
    print ("Numpy installed, version", np.__version__)
except ImportError:
    print ("Numpy is not installed!")
    errors += 1

try:
    import scipy
    print ("Scipy installed, version", scipy.__version__)
except ImportError:
    print ("Scipy is not installed!")
    errors += 1

try:
    import matplotlib
    print ("Matplotlib installed, version", matplotlib.__version__)
except ImportError:
    print ("Matplotlib is not installed!")
    errors += 1

try:
    import sklearn
    print ("Sklearn installed, version", sklearn.__version__)
except ImportError:
    print ("Sklearn is not installed!")
    errors += 1

try:
    import bs4
    print ("Beautifulsoup4  installed, version", bs4.__version__)
except ImportError:
    print ("Beautifulsoup4  is not installed!")
    errors += 1

try:
    import networkx
    print ("Networkx installed, version", networkx.__version__)
except ImportError:
    print ("Networkx is not installed!")
    errors += 1

try:
    import nltk
    print ("Nltk installed, version", nltk.__version__)
except ImportError:
    print ("Nltk is not installed!")
    errors += 1

try:
    import gensim
    print ("Gensim installed, version", gensim.__version__)
except ImportError:
    print ("Gensim is not installed!")
    errors += 1

try:
    import xgboost
    print ("XGBoost installed, version", xgboost.__version__)
except ImportError:
    print ("XGBoost is not installed!")
    errors += 1

try:
    import theano
    print ("Theano installed, version", theano.__version__)
except ImportError:
    print ("Theano is not installed!")
    errors += 1

try:
    import keras
    import pkg_resources
    vs = pkg_resources.get_distribution("keras").version
    print ("Keras installed, version", vs)
except ImportError:
    print ("Keras is not installed!")
    errors += 1

get_ipython().run_cell_magic('latex', '', '\\[\n |u(t)| = \n  \\begin{cases} \n   u(t) & \\text{if } t \\geq 0 \\\\\n   -u(t)       & \\text{otherwise }\n  \\end{cases}\n\\]')

get_ipython().run_cell_magic('latex', '', '\\begin{align}\nf(x) &= (a+b)^2 \\\\\n     &= a^2 + (a+b) + (a+b) + b^2 \\\\\n     &= a^2 + 2\\cdot (a+b) + b^2\n\\end{align}')

if errors == 0:
    print ("Your machine can run the code")
else:
    print ("We found %i errors. Please check them and install the missing toolkits" % errors)

