import os.path as op
from stormdb.process import MNEPython
from stormdb.base import mkdir_p  # mkdir -p

proj_name = 'MINDLAB2015_MEG-Sememene'
subject = '0010_HXN'  # demo subject
scratch_dir = op.join('/projects', proj_name, 'scratch')
subjects_dir = op.join(scratch_dir, 'fs_subjects_dir')

fwd_dir = op.join(scratch_dir, 'fwd_operators')
mkdir_p(fwd_dir)  # create, if not exists

bem_fname = op.join(subjects_dir, subject, 'bem', subject + '-3LBEM-sol.fif')
conductivity = [0.3, 0.006, 0.3]  # brain, skull, skin
# conductivity = [0.3]  # for a single-layer (inner skull) model

mp = MNEPython(proj_name)

mp.prepare_bem_model(subject, bem_fname, subjects_dir=subjects_dir,
                     conductivity=conductivity)

src_fname = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')

mp.setup_source_space(subject, src_fname, subjects_dir=subjects_dir, add_dist=True)

mp.submit()

mp.status

trans_dir = op.join(scratch_dir, 'trans')
trans_fname = op.join(trans_dir, '0010_HXN_20151201-trans.fif')  # example case

meas_fname = op.join(scratch_dir, 'maxfilter/tsss_st10_corr98/0010', 'MMN_block1_raw_tsss.fif')

fwd_fname = op.join(fwd_dir, '0010_HXN-MMN_block1-fwd.fif')

mp_fwd = MNEPython(proj_name)  # we could re-use the mp-object from before
mp_fwd.make_forward_solution(meas_fname, trans_fname, bem_fname, src_fname, fwd_fname)

mp_fwd.submit()

mp_fwd.status

get_ipython().magic('matplotlib qt')

from mne import sensitivity_map, read_forward_solution
fwd = read_forward_solution(fwd_fname, surf_ori=True)

grad_map = sensitivity_map(fwd, ch_type='grad', mode='fixed')

brain = grad_map.plot(time_label='Gradiometer sensitivity', subjects_dir=subjects_dir,
                      clim=dict(lims=[0, 50, 100]))

