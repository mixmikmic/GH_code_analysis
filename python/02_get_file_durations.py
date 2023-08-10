get_ipython().magic('matplotlib inline')
import sox
import pandas as pd
import numpy as np
import json
import pickle
import os
import glob
from tqdm import tqdm

testfolder = '/beegfs/js7561/datasets/dcase2017/task4_official/test/audio/'
trainfolder = '/beegfs/js7561/datasets/dcase2017/task4_official/train/audio/'
evalfolder = '/beegfs/js7561/datasets/dcase2017/task4_official/eval/audio/'

durdict = {}

for f in tqdm(glob.glob(os.path.join(testfolder, '*.wav'))):
    fname = os.path.basename(f).replace('.wav', '')
    duration = sox.file_info.duration(f)
    durdict[fname] = duration

for f in tqdm(glob.glob(os.path.join(trainfolder, '*.wav'))):
    fname = os.path.basename(f).replace('.wav', '')
    duration = sox.file_info.duration(f)
    durdict[fname] = duration

for f in tqdm(glob.glob(os.path.join(evalfolder, '*.wav'))):
    fname = os.path.basename(f).replace('.wav', '')
    duration = sox.file_info.duration(f)
    durdict[fname] = duration

# Save this dictionary to disk
duration_file = '/beegfs/js7561/datasets/dcase2017/task4_official/combined/metadata/durations.json'
with open(duration_file, 'w') as fp:
    json.dump(durdict, fp, indent=2)



