import jams
import os
import sox
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

jams.__version__

DCASE_CLASSES = ['Train horn',
                 'Air horn, truck horn',
                 'Car alarm',
                 'Reversing beeps',
                 'Ambulance (siren)',
                 'Police car (siren)',
                 'Fire engine, fire truck (siren)',
                 'Civil defense siren',
                 'Screaming',
                 'Bicycle',
                 'Skateboard',
                 'Car',
                 'Car passing by',
                 'Bus',
                 'Truck',
                 'Motorcycle',
                 'Train']

def create_dcase_jam(audiofile, weakdf, split, verbose=False):
    
    base_folder = '/beegfs/js7561/datasets/dcase2017/task4_official/'
    splitfolder = os.path.join(base_folder, split)

    # Create jam
    jam = jams.JAMS()

    # Create annotation
    ann = jams.Annotation('tag_open')
    # duration = sox.file_info.duration(audiofile)
    duration = 10.0
    ann.duration = duration
    
    # Get labels from CSV file    
    audiobase = os.path.basename(audiofile)
    fid = audiobase[1:12]
    if verbose:
        print(fid)
    labels = weakdf[weakdf['filename'].str.contains(fid)].label.values
    assert len(labels) > 0
    if verbose:
        print(labels)
        
    # for strong annotation, remove duplicates
    labels = list(np.unique(labels))
    
    # Add tag for each label
    for l in labels:
        ann.append(time=0, duration=duration, value=l, confidence=1)
        
    # Fill file metadata
    jam.file_metadata.title = audiobase
    jam.file_metadata.release = '1.0'
    jam.file_metadata.duration = duration
    jam.file_metadata.artist = ''

    # Fill annotation metadata
    ann.annotation_metadata.version = '1.0'
    ann.annotation_metadata.corpus = 'DCASE 2017 Task 4'
    ann.annotation_metadata.data_source = 'AudioSet'

    # Add annotation to jam
    jam.annotations.append(ann)

    # Return jam
    return jam

# # Test
# jam = create_dcase_jam('Y---lTs1dxhU_30.000_40.000', split='test')
# print(jam)

base_folder = '/beegfs/js7561/datasets/dcase2017/task4_official/'
split = 'eval'

weakcsvfile = os.path.join(base_folder, split, 'annotation_csv', 'groundtruth_strong_label_{:s}uation_set.csv'.format(split))
weakdf = pd.read_csv(weakcsvfile, header=None, sep='\t')
weakdf.columns = ['filename', 'start_time', 'end_time', 'label']

audiofiles = glob.glob(os.path.join(base_folder, split, 'audio_silence', '*.wav'))

verbose=False
for af in tqdm(audiofiles):
# for n, af in enumerate(audiofiles):
#     print(n, os.path.basename(af))
#     if n==20:
#         verbose = True
    jam = create_dcase_jam(af, weakdf, split, verbose=verbose)
    jam.save(af.replace('.wav', '.jams'))



