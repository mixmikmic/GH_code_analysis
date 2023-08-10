import argparse
import sys
import os
import pickle

from tqdm import tqdm
from joblib import Parallel, delayed

from jams.util import smkdirs
from librosa.util import find_files

import milsed.utils

OUTPUT_PATH = os.path.expanduser('~/dev/milsed/models/resources')

DCASE_CLASSES = ['Air horn, truck horn',
                 'Ambulance (siren)',
                 'Bicycle',
                 'Bus',
                 'Car',
                 'Car alarm',
                 'Car passing by',
                 'Civil defense siren',
                 'Fire engine, fire truck (siren)',
                 'Motorcycle',
                 'Police car (siren)',
                 'Reversing beeps',
                 'Screaming',
                 'Skateboard',
                 'Train',
                 'Train horn',
                 'Truck']

def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([os.path.join(outdir, milsed.utils.base(aud)),
                                'h5'])
    milsed.utils.save_h5(fname, **data)

input_path = '/beegfs/js7561/datasets/dcase2017/task4_official/eval/audio_silence/'
output_path = '/beegfs/js7561/datasets/dcase2017/task4_official/eval/features_silence/'
n_jobs = 4
use_tqdm = True
overwrite = True

smkdirs(OUTPUT_PATH)
smkdirs(output_path)

pumpfile = os.path.join(OUTPUT_PATH, 'pump.pkl')
with open(pumpfile, 'rb') as fp:
    pump = pickle.load(fp)

# stream = milsed.utils.get_ann_audio(params.input_path)
stream = find_files(input_path)

if not overwrite:
    missing_stream = []
    for af in stream:
        basename = milsed.utils.base(af)
        pumpfile = os.path.join(output_path, basename + '.h5')
        if not os.path.isfile(pumpfile):
            missing_stream.append(af)
    stream = missing_stream

if use_tqdm:
    stream = tqdm(stream, desc='Converting eval data')
    
Parallel(n_jobs=n_jobs)(delayed(convert)(aud, None, pump, output_path) for aud in stream)

h5file = '/beegfs/js7561/datasets/dcase2017/task4_official/eval/features_silence/Y-0RWZT-miFs_420.000_430.000.h5'

dpump = milsed.utils.load_h5(h5file)
datum = dpump['mel/mag']
ytrue = dpump['static/tags'][0]

ytrue



