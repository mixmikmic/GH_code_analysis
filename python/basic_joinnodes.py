from nipype import JoinNode, Node, Workflow
from nipype.interfaces.utility import Function, IdentityInterface

def get_data_from_id(id):
    """Generate a random number based on id"""
    import numpy as np
    return id + np.random.rand()

def merge_and_scale_data(data2):
    """Scale the input list by 1000"""
    import numpy as np
    return (np.array(data2) * 1000).tolist()


node1 = Node(Function(input_names=['id'],
                      output_names=['data1'],
                      function=get_data_from_id),
             name='get_data')
node1.iterables = ('id', [1, 2, 3])

node2 = JoinNode(Function(input_names=['data2'],
                          output_names=['data_scaled'],
                          function=merge_and_scale_data),
                 name='scale_data',
                 joinsource=node1,
                 joinfield=['data2'])

wf = Workflow(name='testjoin')
wf.connect(node1, 'data1', node2, 'data2')
eg = wf.run()

wf.write_graph(graph2use='exec')
from IPython.display import Image
Image(filename='graph_detailed.dot.png')

res = [node for node in eg.nodes() if 'scale_data' in node.name][0].result
res.outputs

res.inputs

def get_data_from_id(id):
    import numpy as np
    return id + np.random.rand()

def scale_data(data2):
    import numpy as np
    return data2

def replicate(data3, nreps=2):
    return data3 * nreps

node1 = Node(Function(input_names=['id'],
                      output_names=['data1'],
                      function=get_data_from_id),
             name='get_data')
node1.iterables = ('id', [1, 2, 3])

node2 = Node(Function(input_names=['data2'],
                      output_names=['data_scaled'],
                      function=scale_data),
             name='scale_data')

node3 = JoinNode(Function(input_names=['data3'],
                          output_names=['data_repeated'],
                          function=replicate),
                 name='replicate_data',
                 joinsource=node1,
                 joinfield=['data3'])

wf = Workflow(name='testjoin')
wf.connect(node1, 'data1', node2, 'data2')
wf.connect(node2, 'data_scaled', node3, 'data3')
eg = wf.run()

wf.write_graph(graph2use='exec')
Image(filename='graph_detailed.dot.png')

