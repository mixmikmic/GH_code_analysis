import json
import numpy as np
from sklearn.model_selection import GroupKFold, ShuffleSplit
import os
import glob
import librosa

artist_index_simple = json.load(open('/home/bmcfee/git/milsed/models_medley/resources/medley_artist_index.json'))

artist_index = {}

for key in artist_index_simple:
    
    hits = sorted(glob.glob('/home/bmcfee/data/Medleydb_Downmix/{}*.jams'.format(key)))
    new_keys = [os.path.splitext(os.path.basename(_))[0] for _ in hits]
    
    for k in new_keys:
        artist_index[k] = artist_index_simple[key]

durations = {}
for key in artist_index:
    duration = librosa.get_duration(filename='/home/bmcfee/data/Medleydb_Downmix/{}.ogg'.format(key))
    durations[key] = duration

items = sorted(list(artist_index.keys()))

groups = [artist_index[item] for item in items]

G = GroupKFold(n_splits=5)

for fold, (train_full, test) in enumerate(G.split(X=np.arange(len(items)), groups=groups)):
    # This gives us a train-test split.
    # Now we need to make a validate split
    idx_test = [items[t] for t in test]
    S = ShuffleSplit(n_splits=1, test_size=0.25, random_state=20180227)
    for train, val in S.split(X=np.arange(len(train_full)), y=np.zeros(len(train_full))):
        idx_train = [items[train_full[t]] for t in train]
        idx_val = [items[train_full[t]] for t in val]
        
    with open('/home/bmcfee/git/milsed/models_medley/resources/index_train{:02d}.json'.format(fold), 'w') as fdesc:
        json.dump(dict(id=idx_train), fdesc, indent=2)
    with open('/home/bmcfee/git/milsed/models_medley/resources/index_validate{:02d}.json'.format(fold), 'w') as fdesc:
        json.dump(dict(id=idx_val), fdesc, indent=2)
    with open('/home/bmcfee/git/milsed/models_medley/resources/index_test{:02d}.json'.format(fold), 'w') as fdesc:
        json.dump(dict(id=idx_test), fdesc, indent=2)

with open('/home/bmcfee/git/milsed/models_medley/resources/durations.json', 'w') as fdesc:
    json.dump(durations, fdesc, indent=2)

import pandas as pd
import jams
from tqdm import tqdm_notebook as tqdm

MEDLEY_CLASSES = ['drum set',
                  'electric bass',
                  'piano',
                  'male singer',
                  'clean electric guitar',
                  'vocalists',
                  'female singer',
                  'acoustic guitar',
                  'distorted electric guitar',
                  'auxiliary percussion',
                  'double bass',
                  'violin',
                  'cello',
                  'flute']

jamses = sorted(glob.glob('/home/bmcfee/data/Medleydb_Downmix/*.jams'))

records = []
for jf in tqdm(jamses):
    fid = os.path.splitext(os.path.basename(jf))[0]
    jam = jams.load(jf)
    ann = jam.annotations['tag_medleydb_instruments', 0]
    
    for obs in ann:
        if obs.value not in MEDLEY_CLASSES:
            continue
        records.append((fid, obs.time, obs.time + obs.duration, obs.value))

df = pd.DataFrame.from_records(records, columns=['id', 'start', 'end', 'value'])

df.to_csv('/home/bmcfee/git/milsed/models_medley/resources/gt_all.csv', sep='\t', header=False, index=False)

