get_ipython().magic('pylab inline')
from IPython.display import Image
import os.path as op
import nibabel as nb
data_dir = '/home/jovyan/work/data/ds000114/'

def plot_nii(in_file):
    nb.load(in_file).orthoview()

from nipype import MapNode
from nipype.interfaces import fsl

bet_node = MapNode(fsl.BET(), name='bet', iterfield=['in_file'])
bet_node.inputs.in_file = [
    op.join(data_dir, 'sub-01', 'anat', 'sub-01_T1w.nii.gz'),
    op.join(data_dir, 'sub-02', 'anat', 'sub-02_T1w.nii.gz'),
]

res = bet_node.run()
print(res.outputs.out_file)

plot_nii(res.outputs.out_file[0])
plot_nii(res.outputs.out_file[1])

# Solution 1: only MapNode
import glob
mcflirt_node = MapNode(fsl.MCFLIRT(), name='mcflirt', iterfield=['in_file'])
mcflirt_node.base_dir = 'nipype_101'
mcflirt_node.inputs.in_file = glob.glob(op.join(data_dir, 'sub-0[1,2]', 'func', '*_bold.nii.gz'))
res = mcflirt_node.run()
res.outputs.out_file

# Solution 2: iterables + MapNode
from nipype import Workflow, Node
from nipype.interfaces.utility import Function

def _get_bolds(subject_id, data_dir):
    import glob
    import os.path as op
    return glob.glob(op.join(data_dir, 'sub-%s' % subject_id, 'func', '*_bold.nii.gz'))

dg_node = Node(Function(input_names=['subject_id', 'data_dir'], output_names=['out_files'], function=_get_bolds),
               name='datagrabber')
mcflirt_node = MapNode(fsl.MCFLIRT(), name='mcflirt', iterfield=['in_file'])

wf = Workflow('task2')
wf.base_dir = 'workdir_task2'
wf.connect(dg_node, 'out_files', mcflirt_node, 'in_file')

dg_node.iterables = ('subject_id', ['01', '02'])
dg_node.inputs.data_dir = data_dir
wf.run()

from bids.grabbids import BIDSLayout
layout = BIDSLayout(data_dir)
print('Number of subjects is %d' % len(layout.get_subjects()))

def get_info(subject_id, dataset_dir):
    from bids.grabbids import BIDSLayout
    layout = BIDSLayout(dataset_dir)
    t1w = layout.get(type='T1w', 
                     subject=subject_id, 
                     extensions=['nii', 'nii.gz'],
                     return_type='file')
    epi = [f.filename for f in layout.get(type='bold', subject=subject_id, extensions=['nii', 'nii.gz'])]
    return t1w, epi

# Test get_info:
print(get_info('01', data_dir))
print(get_info('0[12]', data_dir))

node_bids = Node(Function(input_names=['subject_id', 'dataset_dir'], output_names=['t1w', 'epi'],
                          function=get_info), name='bids_ds')
node_bids.inputs.subject_id = '0[12]'
node_bids.inputs.dataset_dir = data_dir

# Test this node
res = node_bids.run()
print(res.outputs.t1w)
print(res.outputs.epi)

# We will need this little function
def _flatten(inlist):
    return [el for l in inlist for el in l]

print(_flatten([[1, 2], [3], [4, 5]]))

def _vol_mean(in_file):
    import nibabel as nb
    import numpy as np
    nii = nb.load(in_file)
    
    data = nii.get_data()
    voxvol = np.prod(nii.header.get_zooms()[:3])
    volume = sum(data > 0) * voxvol
    
    mean = data[data > 0].mean()
    
    return volume, mean

from nipype.interfaces.utility import IdentityInterface, Rename
from nipype.interfaces.io import DataSink

inputs_node = Node(IdentityInterface(fields=['subject_id']), name='inputnode')
bids_node = MapNode(Function(input_names=['subject_id', 'dataset_dir'], output_names=['t1w', 'epi'],
                             function=get_info),
                    name='bids_ds', 
                    iterfield=['subject_id']
)
bet_node = MapNode(fsl.BET(functional=True, mask=True), name='bet', iterfield=['in_file'])
mcflirt_node = MapNode(fsl.MCFLIRT(), name='mcflirt', iterfield=['in_file'])
applymsk_node = MapNode(fsl.ApplyMask(), name='mask', iterfield=['in_file', 'mask_file'])

volmean_node = MapNode(Function(input_names=['in_file'], output_names=['vol', 'mean'],
                       function=_vol_mean), name='volmean', iterfield=['in_file'])

ds_node = Node(DataSink(
    base_directory='/home/jovyan/work/workshops/170327-'
                   'nipype/notebooks/nipype_101_tasks/nipype_outputs'),
               name='datasink')

wf = Workflow('task_5')
wf.base_dir = 'nipypework'
wf.connect(inputs_node, 'subject_id', bids_node, 'subject_id')
wf.connect(bids_node, ('epi', _flatten), bet_node, 'in_file')
wf.connect(bids_node, ('epi', _flatten), mcflirt_node, 'in_file')
wf.connect(bet_node, 'mask_file', applymsk_node, 'mask_file')
wf.connect(mcflirt_node, 'out_file', applymsk_node, 'in_file')
wf.connect(applymsk_node, 'out_file', ds_node, 'epi')
wf.connect(applymsk_node, 'out_file', volmean_node, 'in_file')


Image(wf.write_graph())

# Time to run it
wf.inputs.inputnode.subject_id = ['01', '02']
wf.inputs.bids_ds.dataset_dir = data_dir
wf.run()

get_ipython().system('tree nipype_outputs/')

from nipype import JoinNode

from nipype.interfaces.utility import Merge

get_ipython().magic('pinfo2 Merge')



