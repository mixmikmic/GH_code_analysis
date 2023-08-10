#%% libraries
import os
import sys
import glob
import io
import itertools
import textract
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

#%% reader functions
def pdf_to_txt(inpath, outpath):
    try:
        text = textract.process(inpath, method='pdftotext')
        base = os.path.abspath(inpath)
        wdir, fname = outpath, os.path.split(base)[1]
        writepath = wdir + '/' + fname.split('.')[0] + '.txt'

        with open(writepath, 'wb') as f:
            f.write(text)
    except:
        print(inpath, ' failed')
        pass
    
    
def read_pdf(inpath):
    text = textract.process(inpath, method='pdftotext')
    return text

import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

root = '/media/alal/LAL_DATA/Newspapers/The Kathmandu Post'
os.chdir(root)

#%% directories
input = root 
output = root + '/raw_txts/'

if not os.path.exists(output):
    os.makedirs(output)

get_ipython().magic('pwd ()')

pdfs = []
sizes = {}

for root, dirs, files in os.walk(input):
    for file in files:
        if file.endswith(".pdf") and file[0] != '.':
            ff = os.path.join(root, file)
            pdfs.append(ff)
            size = os.path.getsize(ff) # in bytes
            sizes[file] = size

ser = pd.Series(sizes)
ser.plot.density()
convert_size(ser.min())
convert_size(ser.max())

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

get_ipython().run_cell_magic('time', '', 'results = Parallel(n_jobs=num_cores)(delayed(pdf_to_txt)(p,output) \\\n                                     for p in pdfs)')

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

# pick file, remove punctuation and stopwords
tmp = '/home/alal/tmp'
inp = root + 'raw_txts'
out = root + '/sentences/'

if not os.path.exists(out):
    os.makedirs(out)

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

files = glob.glob(inp+'/TKP_*.txt')

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

get_ipython().run_cell_magic('time', '', 'results = Parallel(n_jobs=num_cores)(delayed(write_word_freqs)(i,out) \\\n                                     for i in files)')

files = glob.glob(inp+'/TKP_*.txt')

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

get_ipython().run_cell_magic('time', '', 'results = Parallel(n_jobs=num_cores)(delayed(write_sentences)(i,out) \\\n                                     for i in files)')

