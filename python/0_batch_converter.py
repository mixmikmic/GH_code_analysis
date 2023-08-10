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
        print(inpath, ' has incompatible characters. Run again')
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

root = '/media/alal/LAL_DATA/Newspapers/The Himalayan Times'
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



