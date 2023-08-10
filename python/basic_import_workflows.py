from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
smoothwf = create_susan_smooth()

get_ipython().system('fslmaths /data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz     -Tmean -thrP 50 /output/sub-01_ses-test_task-fingerfootlips_mask.nii.gz')

smoothwf.inputs.inputnode.in_files = '/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz'
smoothwf.inputs.inputnode.mask_file = '/output/sub-01_ses-test_task-fingerfootlips_mask.nii.gz'
smoothwf.inputs.inputnode.fwhm = 4
smoothwf.base_dir = '/output'

get_ipython().magic('pylab inline')
from IPython.display import Image
smoothwf.write_graph(graph2use='colored', format='png', simple_form=True)
Image(filename='/output/susan_smooth/graph.dot.png')

smoothwf.run('MultiProc', plugin_args={'n_procs': 4})

get_ipython().system('fslmaths /data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz -Tmean fmean.nii.gz')
get_ipython().system('fslmaths /output/susan_smooth/smooth/mapflow/_smooth0/sub-01_ses-test_task-fingerfootlips_bold_smooth.nii.gz     -Tmean smean.nii.gz')

from nilearn import image, plotting
plotting.plot_epi(
    'fmean.nii.gz', title="mean (no smoothing)", display_mode='z',
    cmap='gray', cut_coords=(-45, -30, -15, 0, 15))
plotting.plot_epi(
    'smean.nii.gz', title="mean (susan smoothed)", display_mode='z',
    cmap='gray', cut_coords=(-45, -30, -15, 0, 15))

print(smoothwf.list_node_names())

median = smoothwf.get_node('median')

median.inputs.op_string = '-k %s -p 99'

smoothwf.run('MultiProc', plugin_args={'n_procs': 4})

get_ipython().system('fslmaths /output/susan_smooth/smooth/mapflow/_smooth0/sub-01_ses-test_task-fingerfootlips_bold_smooth.nii.gz     -Tmean mmean.nii.gz')

from nilearn import image, plotting
plotting.plot_epi(
    'smean.nii.gz', title="mean (susan smooth)", display_mode='z',
    cmap='gray', cut_coords=(-45, -30, -15, 0, 15))
plotting.plot_epi(
    'mmean.nii.gz', title="mean (smoothed, median=99%)", display_mode='z',
    cmap='gray', cut_coords=(-45, -30, -15, 0, 15))

