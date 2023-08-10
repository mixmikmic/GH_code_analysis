import numpy as np
import pylab as plt
import pymzml
get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import sys
sys.path.append('/Users/simon/git/lda/code/')
from lda import VariationalLDA

prefix = '/Users/simon/Dropbox/MS2LDA Manuscript Sections/Matrices/Beer3pos_MS1filter_Method3'

v_lda = VariationalLDA(K = 300,alpha = 1,eta=0.1,update_alpha=True)

v_lda.load_features_from_csv(prefix,scale_factor=100.0)

v_lda.run_vb(n_its = 10,initialise=True)

v_lda.run_vb(n_its = 10,initialise=False)

t = 3
td = v_lda.get_topic_as_dict(t)
print td

eth = v_lda.get_expect_theta()
print eth.shape
for doc in v_lda.corpus:
    print "Document: " + str(doc)
    doc_pos = v_lda.doc_index[doc] # This is its row in eth
    for k in range(v_lda.K):
        if eth[doc_pos,k] > 0.01:
            print "{} : {}".format(k,eth[doc_pos,k])
    # Break the loop so as not to get all output
    break



