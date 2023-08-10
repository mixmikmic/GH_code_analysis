#%% libraries
import os
import sys
import glob
import io
import itertools
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().magic('matplotlib inline')

# run for jupyter notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

root = '/media/alal/LAL_DATA/Newspapers/The Himalayan Times/'
os.chdir(root)
get_ipython().magic('pwd ()')

# pick file, remove punctuation and stopwords
tmp = '/home/alal/tmp'
inp = root + '/raw_txts'
out = root + '/word_frequencies/'

if not os.path.exists(out):
    os.makedirs(out)

def write_word_freqs(inputfile,outdir):
    filterout= set(stopwords.words('english')+
               list(string.punctuation)+
               ['\'\'','``','\'s','’',"“","”",
                'the','said','nepal','world','kathmandu'])
    cols = ['word','freq']

    base = os.path.abspath(inputfile)
    wdir, fname = outdir, os.path.split(base)[1]
    writepath = wdir + '/wfreqs_' + fname.split('.')[0] + '.csv'

    f = open(inputfile)
    raw = f.read()
    tokens = [token.lower() for token in nltk.word_tokenize(raw)]
    cleaned = [token for token in tokens if token not in filterout]
    
    fdict = dict(nltk.FreqDist(cleaned))
    df = pd.DataFrame(list(fdict.items()),columns=cols)
    df = df.sort_values('freq',ascending=0)
    
    df.to_csv(writepath,columns=['word','freq'])

nltk.data.path.append('/media/alal/LAL_DATA/Newspapers/nltk_data')

def write_sentences(inputfile,outdir):
    base = os.path.abspath(inputfile)
    wdir, fname = outdir, os.path.split(base)[1]
    writepath = wdir + '/sentences_' + fname.split('.')[0] + '.txt'

    f = open(inputfile)
    raw = f.read()
    string = raw.replace('\n'," ")
    sentences = [token.lower() for token in nltk.tokenize.sent_tokenize(string)]

    outF = open(writepath, "w")
    sentences = map(lambda x: x+"\n", sentences)

    outF.writelines(sentences)
    outF.close()

# pick file, remove punctuation and stopwords
tmp = '/home/alal/tmp'
inp = root + 'raw_txts'
out = root + '/word_frequencies/'

if not os.path.exists(out):
    os.makedirs(out)

files = glob.glob(inp+'/THT_*.txt')

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

get_ipython().run_cell_magic('time', '', 'results = Parallel(n_jobs=num_cores)(delayed(write_word_freqs)(i,out) \\\n                                     for i in files)')

# pick file, remove punctuation and stopwords
tmp = '/home/alal/tmp'
inp = root + 'raw_txts'
out = root + '/sentences/'

if not os.path.exists(out):
    os.makedirs(out)

files = glob.glob(inp+'/THT_*.txt')

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

get_ipython().run_cell_magic('time', '', 'results = Parallel(n_jobs=num_cores)(delayed(write_sentences)(i,out) \\\n                                     for i in files)')



