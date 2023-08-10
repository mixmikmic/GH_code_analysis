from mne.io import read_raw_fif, RawArray, read_info
from mne.io.constants import FIFF
from stormdb.access import Query
from stormdb.base import mkdir_p  # mkdir -p
import os.path as op
import numpy as np

proj_name = 'MINDLAB2015_MEG-Sememene'
trans_dir = op.join('/projects', proj_name, 'scratch', 'trans')
mkdir_p(trans_dir)  # this must exist

qy = Query(proj_name)
included_subjects = qy.get_subjects()

for sub in included_subjects:
    studies = qy.get_studies(sub, modality='MEG')
    for study in studies:
        study_date = study[:8]  # cut out the '_000000'
        output_fname = '_'.join([sub, study_date, 'coreg.fif'])

        series = qy.filter_series('*', subjects=sub,
                                  study_date_range=study_date)
        # any file from this study will do, use the first one
        raw_fname = op.join(series[0]['path'], series[0]['files'][0])
        
        info = read_info(raw_fname)
        # do an in-place replacement, consider making a copy and
        # adding the EEG points as EXTRA points to the end?
        for digpoint in info['dig']:
            if digpoint['kind'] == FIFF.FIFFV_POINT_EEG:
                digpoint['kind'] = FIFF.FIFFV_POINT_EXTRA
        raw = RawArray(np.empty((info['nchan'], 1)), info)
        raw.save(op.join(trans_dir, output_fname), overwrite=True) 

