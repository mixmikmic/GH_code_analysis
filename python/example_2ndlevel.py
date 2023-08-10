get_ipython().magic('matplotlib inline')
from os.path import join as opj
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import (OneSampleTTestDesign, EstimateModel,
                                   EstimateContrast, Threshold)
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.fsl import Info
from nipype.algorithms.misc import Gunzip

experiment_dir = '/output'
output_dir = 'datasink'
working_dir = 'workingdir'

# Smoothing withds used during preprocessing
fwhm = [4, 8]

# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006', 'con_0007']

mask = "/data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c/2mm_brainmask.nii.gz"

# Gunzip - unzip the mask image
gunzip = Node(Gunzip(in_file=mask), name="gunzip")

# OneSampleTTestDesign - creates one sample T-Test Design
onesamplettestdes = Node(OneSampleTTestDesign(),
                         name="onesampttestdes")

# EstimateModel - estimates the model
level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Threshold - thresholds contrasts
level2thresh = Node(Threshold(contrast_index=1,
                              use_topo_fdr=True,
                              use_fwe_correction=False,
                              extent_threshold=0,
                              height_threshold=0.005,
                              height_threshold_type='p-value',
                              extent_fdr_p_threshold=0.05),
                    name="level2thresh")

# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['contrast_id', 'fwhm_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list),
                        ('fwhm_id', fwhm)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'cons': opj(output_dir, 'norm_spm', 'sub-*_fwhm{fwhm_id}',
                         'w{contrast_id}.nii')}
selectfiles = Node(SelectFiles(templates,
                               base_directory=experiment_dir,
                               sort_filelist=True),
                   name="selectfiles")

# Datasink - creates output folder for important outputs
datasink = Node(DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")

# Use the following DataSink output substitutions
substitutions = [('_contrast_id_', '')]
subjFolders = [('%s_fwhm_id_%s' % (con, f), 'spm_%s_fwhm%s' % (con, f))
               for f in fwhm
               for con in contrast_list]
substitutions.extend(subjFolders)
datasink.inputs.substitutions = substitutions

# Initiation of the 2nd-level analysis workflow
l2analysis = Workflow(name='l2analysis')
l2analysis.base_dir = opj(experiment_dir, working_dir)

# Connect up the 2nd-level analysis components
l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ('fwhm_id', 'fwhm_id')]),
                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    (gunzip, onesamplettestdes, [('out_file',
                                                  'explicit_mask_file')]),
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                                                    '2ndLevel.@spm_mat'),
                                                   ('spmT_images',
                                                    '2ndLevel.@T'),
                                                   ('con_images',
                                                    '2ndLevel.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel.@threshold')]),
                    ])

# Create 1st-level analysis output graph
l2analysis.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=opj(l2analysis.base_dir, 'l2analysis', 'graph.dot.png'))

l2analysis.run('MultiProc', plugin_args={'n_procs': 4})

# Change the SelectFiles template and recreate the node
templates = {'cons': opj(output_dir, 'norm_ants', 'sub-*_fwhm{fwhm_id}',
                         '{contrast_id}_trans.nii')}
selectfiles = Node(SelectFiles(templates,
                               base_directory=experiment_dir,
                               sort_filelist=True),
                   name="selectfiles")

# Change the substituion parameters for the datasink
substitutions = [('_contrast_id_', '')]
subjFolders = [('%s_fwhm_id_%s' % (con, f), 'ants_%s_fwhm%s' % (con, f))
               for f in fwhm
               for con in contrast_list]
substitutions.extend(subjFolders)
datasink.inputs.substitutions = substitutions

# Initiation of the 2nd-level analysis workflow
l2analysis = Workflow(name='l2analysis')
l2analysis.base_dir = opj(experiment_dir, working_dir)

# Connect up the 2nd-level analysis components
l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ('fwhm_id', 'fwhm_id')]),
                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    (gunzip, onesamplettestdes, [('out_file',
                                                  'explicit_mask_file')]),
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                                                    '2ndLevel.@spm_mat'),
                                                   ('spmT_images',
                                                    '2ndLevel.@T'),
                                                   ('con_images',
                                                    '2ndLevel.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel.@threshold')]),
                    ])

l2analysis.run('MultiProc', plugin_args={'n_procs': 4})

get_ipython().magic('pylab inline')
from nilearn.plotting import plot_stat_map
anatimg = '/data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c/1mm_T1.nii.gz'

plot_stat_map(
    '/output/datasink/2ndLevel/ants_con_0001_fwhm4/spmT_0001_thr.nii', title='ants fwhm=4',
    bg_img=anatimg, threshold=3, vmax=10, display_mode='y', cut_coords=(-30, -15, 0, 15, 30), cmap='viridis', dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/ants_con_0001_fwhm8/spmT_0001_thr.nii', title='ants fwhm=8',
    bg_img=anatimg, threshold=3, vmax=10, display_mode='y', cut_coords=(-30, -15, 0, 15, 30), cmap='viridis', dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/spm_con_0001_fwhm4/spmT_0001_thr.nii', title='spm fwhm=4',
    bg_img=anatimg, threshold=3, vmax=10, display_mode='y', cut_coords=(-30, -15, 0, 15, 30), cmap='viridis', dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/spm_con_0001_fwhm8/spmT_0001_thr.nii', title='spm fwhm=8',
    bg_img=anatimg, threshold=3, vmax=10, display_mode='y', cut_coords=(-30, -15, 0, 15, 30), cmap='viridis', dim=1)

from nilearn.plotting import plot_stat_map
anatimg = '/data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c/1mm_T1.nii.gz'

plot_stat_map(
    '/output/datasink/2ndLevel/ants_con_0002_fwhm4/spmT_0001_thr.nii', title='ants fwhm=4',
    bg_img=anatimg, threshold=3.7, vmax=10, cmap='viridis', display_mode='y', cut_coords=(-30, -15, 0, 15, 30), dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/ants_con_0002_fwhm8/spmT_0001_thr.nii', title='ants fwhm=8',
    bg_img=anatimg, threshold=3.7, vmax=10, cmap='viridis', display_mode='y', cut_coords=(-30, -15, 0, 15, 30), dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/spm_con_0002_fwhm4/spmT_0001_thr.nii', title='spm fwhm=4',
    bg_img=anatimg, threshold=3.7, vmax=10, cmap='viridis', display_mode='y', cut_coords=(-30, -15, 0, 15, 30), dim=1)

plot_stat_map(
    '/output/datasink/2ndLevel/spm_con_0002_fwhm8/spmT_0001_thr.nii', title='spm fwhm=8',
    bg_img=anatimg, threshold=3.7, vmax=10, cmap='viridis', display_mode='y', cut_coords=(-30, -15, 0, 15, 30), dim=1)

from nilearn.plotting import plot_glass_brain
plot_glass_brain(
    '/output/datasink/2ndLevel/spm_con_0002_fwhm4/spmT_0001_thr.nii',
    threshold=3.7, display_mode='lyrz', black_bg=True, vmax=10, title='spm_fwhm4')
plot_glass_brain(
    '/output/datasink/2ndLevel/spm_con_0002_fwhm8/spmT_0001_thr.nii',
    threshold=3.7, display_mode='lyrz', black_bg=True, vmax=10, title='spm_fwhm8')
plot_glass_brain(
    '/output/datasink/2ndLevel/ants_con_0002_fwhm4/spmT_0001_thr.nii',
    threshold=3.7, display_mode='lyrz', black_bg=True, vmax=10, title='ants_fwhm4')
plot_glass_brain(
    '/output/datasink/2ndLevel/ants_con_0002_fwhm8/spmT_0001_thr.nii',
    threshold=3.7, display_mode='lyrz', black_bg=True, vmax=10, title='ants_fwhm8')

