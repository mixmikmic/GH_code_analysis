from nipype.caching import Memory
mem = Memory(base_dir='/output/workingdir')

from nipype.interfaces import fsl
bet_mem = mem.cache(fsl.BET)

bet_mem(in_file="/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz",
        out_file="/output/sub-02_T1w_brain.nii.gz",
        mask=True)

get_ipython().system(' ls -l /output/workingdir/nipype_mem')

bet_mem(in_file="/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz",
        out_file="/output/sub-02_T1w_brain.nii.gz",
        mask=True)

mem.clear_runs_since()
bet_mem(in_file="/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz",
        out_file="/output/sub-02_T1w_brain.nii.gz",
        mask=True)

mem.clear_runs_since(year=2020, month=1, day=1)

