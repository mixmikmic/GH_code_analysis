from nipype import SelectFiles, Node, Workflow
from os.path import abspath as opap
from nipype.interfaces.fsl import MCFLIRT, IsotropicSmooth

# Create SelectFiles node
templates={'func': '{subject_id}/ses-test/func/{subject_id}_ses-test_task-fingerfootlips_bold.nii.gz'}
sf = Node(SelectFiles(templates),
          name='selectfiles')
sf.inputs.base_directory = opap('/data/ds000114')
sf.inputs.subject_id = 'sub-11'

# Create Motion Correction Node
mcflirt = Node(MCFLIRT(mean_vol=True,
                       save_plots=True),
               name='mcflirt')

# Create Smoothing node
smooth = Node(IsotropicSmooth(fwhm=4),
              name='smooth')

# Create a preprocessing workflow
wf = Workflow(name="preprocWF")
wf.base_dir = 'working_dir'

# Connect the three nodes to each other
wf.connect([(sf, mcflirt, [("func", "in_file")]),
            (mcflirt, smooth, [("out_file", "in_file")])])

# Let's the workflow
wf.run()

get_ipython().system('nipypecli crash /opt/tutorial/notebooks/crash-*selectfiles-*.pklz')

get_ipython().system('nipypecli crash -r /opt/tutorial/notebooks/crash-*selectfiles-*.pklz')

wf.config['execution']['crashfile_format'] = 'txt'
wf.run()

get_ipython().system('cat /opt/tutorial/notebooks/crash-*selectfiles-*.txt')

from nipype.interfaces.fsl import IsotropicSmooth
smooth = IsotropicSmooth(fwhm='4')

IsotropicSmooth.help()

from nipype.interfaces.fsl import IsotropicSmooth
smooth = IsotropicSmooth(output_type='NIFTIiii')

from nipype.algorithms.misc import Gunzip
from nipype.pipeline.engine import Node

files = ['/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz',
         '/data/ds000114/sub-02/ses-test/func/sub-02_ses-test_task-fingerfootlips_bold.nii.gz']

gunzip = Node(Gunzip(), name='gunzip',)
gunzip.inputs.in_file = files

from nipype.pipeline.engine import MapNode
gunzip = MapNode(Gunzip(), name='gunzip', iterfield=['in_file'])
gunzip.inputs.in_file = files

files = ['/data/ds000114/sub-01/func/sub-01_task-fingerfootlips_bold.nii.gz',
         '/data/ds000114/sub-03/func/sub-03_task-fingerfootlips_bold.nii.gz']
gunzip.inputs.in_file = files

from nipype.interfaces.spm import Smooth
smooth = Smooth(in_files='/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz')
smooth.run()

from nipype.interfaces.spm import Realign
realign = Realign(in_files='/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz')
realign.run()

from nipype.interfaces.spm import Realign
realign = Realign(register_to_mean=True)
realign.run()

realign.help()

from nipype.interfaces.afni import Despike
despike = Despike(in_file='/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz',
                  output_type='NIFTI')
despike.run()

from nipype import SelectFiles, Node, Workflow
from os.path import abspath as opap
from nipype.interfaces.fsl import MCFLIRT, IsotropicSmooth

# Create SelectFiles node
templates={'func': '{subject_id}/func/{subject_id}_task-fingerfootlips_bold.nii.gz'}
sf = Node(SelectFiles(templates),
          name='selectfiles')
sf.inputs.base_directory = opap('/data/ds000114')
sf.inputs.subject_id = 'sub-01'

# Create Motion Correction Node
mcflirt = Node(MCFLIRT(mean_vol=True,
                       save_plots=True),
               name='mcflirt')

# Create Smoothing node
smooth = Node(IsotropicSmooth(fwhm=4),
              name='smooth')

# Create a preprocessing workflow
wf = Workflow(name="preprocWF")
wf.base_dir = 'working_dir'

# Connect the three nodes to each other
wf.connect([(sf, mcflirt, [("func", "in_file")]),
            (mcflirt, smooth, [("out_file", "in_file")])])

# Create a new node
mcflirt_NEW = Node(MCFLIRT(mean_vol=True),
                   name='mcflirt_NEW')

# Connect it to an already connected input field
wf.connect([(mcflirt_NEW, smooth, [("out_file", "in_file")])])

