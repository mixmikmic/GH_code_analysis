from nipype import SelectFiles, Node

# Create SelectFiles node
templates={'func': '{subject}/{session}/func/{subject}_{session}_task-fingerfootlips_bold.nii.gz'}
sf = Node(SelectFiles(templates),
          name='selectfiles')
sf.inputs.base_directory = '/data/ds000114'
sf.inputs.subject = 'sub-01'
sf.inputs.session = 'ses-test'

from nipype.interfaces.fsl import MCFLIRT, IsotropicSmooth

# Create Motion Correction Node
mcflirt = Node(MCFLIRT(mean_vol=True,
                       save_plots=True),
               name='mcflirt')

# Create Smoothing node
smooth = Node(IsotropicSmooth(fwhm=4),
              name='smooth')

from nipype import Workflow
from os.path import abspath

# Create a preprocessing workflow
wf = Workflow(name="preprocWF")
wf.base_dir = '/output/working_dir'

# Connect the three nodes to each other
wf.connect([(sf, mcflirt, [("func", "in_file")]),
            (mcflirt, smooth, [("out_file", "in_file")])])

wf.run()

get_ipython().system(' tree /output/working_dir/preprocWF')

from nipype.interfaces.io import DataSink

# Create DataSink object
sinker = Node(DataSink(), name='sinker')

# Name of the output folder
sinker.inputs.base_directory = '/output/working_dir/preprocWF_output'

# Connect DataSink with the relevant nodes
wf.connect([(smooth, sinker, [('out_file', 'in_file')]),
            (mcflirt, sinker, [('mean_img', 'mean_img'),
                               ('par_file', 'par_file')]),
            ])
wf.run()

get_ipython().system(' tree /output/working_dir/preprocWF_output')

wf.connect([(smooth, sinker, [('out_file', 'preproc.@in_file')]),
            (mcflirt, sinker, [('mean_img', 'preproc.@mean_img'),
                               ('par_file', 'preproc.@par_file')]),
            ])
wf.run()

get_ipython().system(' tree /output/working_dir/preprocWF_output')

# Define substitution strings
substitutions = [('_task-fingerfootlips', ''),
                 ("_ses-test", ""),
                 ('_bold_mcf', ''),
                 ('.nii.gz_mean_reg', '_mean'),
                 ('.nii.gz.par', '.par')]

# Feed the substitution strings to the DataSink node
sinker.inputs.substitutions = substitutions

# Run the workflow again with the substitutions in place
wf.run()

get_ipython().system(' tree /output/working_dir/preprocWF_output')

