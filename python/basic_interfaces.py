get_ipython().magic('pylab inline')
from nilearn.plotting import plot_anat
plot_anat('/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz', title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

get_ipython().run_cell_magic('bash', '', '\nFILENAME=/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w\n\nbet ${FILENAME}.nii.gz /output/sub-01_ses-test_T1w_bet.nii.gz')

plot_anat('/output/sub-01_ses-test_T1w_bet.nii.gz', title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

get_ipython().run_cell_magic('bash', '', 'bet')

get_ipython().run_cell_magic('bash', '', '\nFILENAME=/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w\n\nbet ${FILENAME}.nii.gz /output/sub-01_ses-test_T1w_bet.nii.gz -m')

plot_anat('/output/sub-01_ses-test_T1w_bet_mask.nii.gz', title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

from nipype.interfaces.fsl import BET

skullstrip = BET()
skullstrip.inputs.in_file = "/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
skullstrip.inputs.out_file = "/output/T1w_nipype_bet.nii.gz"
res = skullstrip.run()

plot_anat('/output/T1w_nipype_bet.nii.gz', title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

print(skullstrip.cmdline)

skullstrip = BET(in_file="/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
                 out_file="/output/T1w_nipype_bet.nii.gz",
                 mask=True)
res = skullstrip.run()

plot_anat('/output/T1w_nipype_bet_mask.nii.gz', title='original',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)

BET.help()

print(res.outputs.mask_file)

skullstrip2 = BET()
skullstrip2.run()

skullstrip.inputs.mask = "mask_file.nii"

skullstrip.inputs.in_file = "/data/oops_a_typo.nii"

skullstrip = BET(in_file="/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")
print(skullstrip.cmdline)

res = skullstrip.run()
print(res.outputs)

res2 = skullstrip.run(mask=True)
print(res2.outputs)

