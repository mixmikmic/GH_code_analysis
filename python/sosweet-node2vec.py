import logging

# utilities and plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# node2vec stuff
import jwalk
from gensim.models import Word2Vec

from nw2vec import utils

logging.basicConfig(level=logging.INFO)

WEIGHTED_NETWORK_FILES = [
    #'dir_weighted_mention_network.txt',
    #'dir_weighted_mention_network_thresh_5.txt',
    'undir_weighted_mention_network_thresh_5.txt',
]

sosweet_embfiles = {}
for netfile in WEIGHTED_NETWORK_FILES:
    sosweet_embfiles[netfile] = utils.node2vec('data/sosweet-network/' + netfile,
                                               'data/sosweet-network/' + netfile[:-4] + '.emb',
                                               num_walks=10, embedding_size=50, window_size=10,
                                               walk_length=80, workers=40, undirected=False)

