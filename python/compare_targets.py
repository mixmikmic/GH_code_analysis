import pandas as pd
import numpy as np
import config
import sys
import re
from ast import literal_eval
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import wordcloud
import matplotlib.pyplot as plt
import matplotlib

get_ipython().magic('matplotlib inline')

pd.set_option('display.max_colwidth', 100)

target_1 = config.target_user
target_2 = config.target_user2
target_month = config.target_month

followers_dir_1 = "{}/data/twitter/followers/{}".format(config.dir_prefix, target_1)
followers_dir_2 = "{}/data/twitter/followers/{}".format(config.dir_prefix, target_2)



