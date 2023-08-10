# Import the function to create an spm fmri preprocessing workflow
from nipype.workflows.fmri.spm import create_spm_preproc

# Create the workflow object
spmflow = create_spm_preproc()

# Import relevant modules
from nipype import IdentityInterface, Node, Workflow

# Create an iternode that iterates over three different fwhm values
inputNode = Node(IdentityInterface(fields=['fwhm']), name='iternode')
inputNode.iterables = ('fwhm', [4, 6, 8])

# Connect inputNode and spmflow in a workflow
metaflow = Workflow(name='metaflow')
metaflow.connect(inputNode, "fwhm", spmflow, "inputspec.fwhm")

# Write graph of type orig
spmflow.write_graph(graph2use='orig', dotfilename='./graph_orig.dot')

# Visulaize graph
from IPython.display import Image
Image(filename="graph_orig.dot.png")

# Write graph of type flat
spmflow.write_graph(graph2use='flat', dotfilename='./graph_flat.dot')

# Visulaize graph
from IPython.display import Image
Image(filename="graph_flat.dot.png")

# Write graph of type hierarchical
metaflow.write_graph(graph2use='hierarchical', dotfilename='./graph_hierarchical.dot')

# Visulaize graph
from IPython.display import Image
Image(filename="graph_hierarchical.dot.png")

# Write graph of type colored
metaflow.write_graph(graph2use='colored', dotfilename='./graph_colored.dot')

# Visulaize graph
from IPython.display import Image
Image(filename="graph_colored.dot.png")

# Write graph of type exec
metaflow.write_graph(graph2use='exec', dotfilename='./graph_exec.dot')

# Visulaize graph
from IPython.display import Image
Image(filename="graph_exec.dot.png")

from IPython.display import Image
Image(filename="graph_flat_detailed.dot.png")

from IPython.display import Image
Image(filename="graph_exec_detailed.dot.png")

# Write graph of type orig
spmflow.write_graph(graph2use='orig', dotfilename='./graph_orig_notSimple.dot', simple_form=False)

# Visulaize graph
from IPython.display import Image
Image(filename="graph_orig_notSimple.dot.png")

