# Import Node and Function module
from nipype import Node, Function

# Create a small example function
def add_two(x_input):
    return x_input + 2

# Create Node
addtwo = Node(Function(input_names=["x_input"],
                       output_names=["val_output"],
                       function=add_two),
              name='add_node')

addtwo.inputs.x_input = 4

addtwo.run()

temp_res = addtwo.run()

temp_res.outputs

addtwo.result.outputs

# Import BET from the FSL interface
from nipype.interfaces.fsl import BET

# Import the Node module
from nipype import Node

# Create Node
bet = Node(BET(frac=0.3), name='bet_node')

# Specify node inputs
bet.inputs.in_file = '/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz'
bet.inputs.out_file = '/output/node_T1w_bet.nii.gz'

res = bet.run()

get_ipython().magic('pylab inline')
from nilearn.plotting import plot_anat
plot_anat(bet.inputs.in_file, title='BET input', cut_coords=(10,10,10),
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
plot_anat(res.outputs.out_file, title='BET output', cut_coords=(10,10,10),
          display_mode='ortho',draw_cross=False, annotate=False)

