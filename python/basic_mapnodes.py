from nipype import Function
def square_func(x):
    return x ** 2
square = Function(["x"], ["f_x"], square_func)

square.run(x=2).outputs.f_x

from nipype import MapNode
square_node = MapNode(square, name="square", iterfield=["x"])

square_node.inputs.x = [0, 1, 2, 3]
square_node.run().outputs.f_x

def power_func(x, y):
    return x ** y

power = Function(["x", "y"], ["f_xy"], power_func)
power_node = MapNode(power, name="power", iterfield=["x", "y"])
power_node.inputs.x = [0, 1, 2, 3]
power_node.inputs.y = [0, 1, 2, 3]
print(power_node.run().outputs.f_xy)

power_node = MapNode(power, name="power", iterfield=["x"])
power_node.inputs.x = [0, 1, 2, 3]
power_node.inputs.y = 3
print(power_node.run().outputs.f_xy)

from nipype.algorithms.misc import Gunzip
from nipype.interfaces.spm import Realign
from nipype.pipeline.engine import Node, MapNode, Workflow

files = ['/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz',
         '/data/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-linebisection_bold.nii.gz']

realign = Node(Realign(register_to_mean=True),
               name='motion_correction')

gunzip = Node(Gunzip(), name='gunzip',)
gunzip.inputs.in_file = files

gunzip = MapNode(Gunzip(), name='gunzip',
                 iterfield=['in_file'])
gunzip.inputs.in_file = files

mcflow = Workflow(name='realign_with_spm')
mcflow.connect(gunzip, 'out_file', realign, 'in_files')
mcflow.base_dir = '/output'
mcflow.run('MultiProc', plugin_args={'n_procs': 4})

