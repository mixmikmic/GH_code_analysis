import numpy as np
import tensorflow as tf

import logging
import sys
root = logging.getLogger()
root.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
root.addHandler(ch)

CLUSTER_SPEC= """
{
    'master' : ['<master-ip>:8888'],
    'ps' : ['<worker0-ip>:8888', '<worker1-ip>:8888'],
    'worker' : [ '<worker2-ip>:8888','<worker3-ip>:8888'],
}
"""
import ast

cluster_def = ast.literal_eval(CLUSTER_SPEC)
spec = tf.train.ClusterSpec(cluster_def)

workers = ['/job:worker/task:{}'.format(i) for i in range(len(cluster_def['worker']))]
param_servers = ['/job:ps/task:{}'.format(i) for i in range(len(cluster_def['ps']))]

batch_size = 1000

graph = tf.Graph()
with graph.as_default():
        
    with tf.device('/job:ps/task:0'):
        input_array = tf.placeholder(tf.int32, shape=[batch_size])
        final_result = tf.Variable(0)
        
    # divide the input across the cluster:
    all_reduce = []
    splitted = tf.split(0, len(workers), input_array)
    for idx, (portion, worker) in enumerate(zip(splitted,workers)):
        with tf.device(worker):
           local_reduce = tf.reduce_sum(portion)
           local_reduce = tf.Print(portion, [local_reduce], message="portion is")
           all_reduce.append(local_reduce)
    
    final_result = tf.reduce_sum(tf.pack(all_reduce))

sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True)

sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True, device_filters=["/job:ps", "/job:worker"])

show_graph(graph)

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

with tf.Session("grpc://tensorflow3.pipeline.io:8888", graph=graph, config=sess_config) as session:
    result = session.run(local_reduce, feed_dict={ input_array: np.ones([1000]) }, options=run_options)
    print(result)

get_ipython().system('ping tensorflow0.pipeline.io')

# TODO:  @Fabrizio, please fill this in...

get_ipython().system('pip install version_information')

get_ipython().run_line_magic('load_ext', 'version_information')
get_ipython().run_line_magic('version_information', 'numpy, scipy, matplotlib, pandas, tensorflow, sklearn, skflow')

import tensorflow as tf
import numpy as np

# Note:  All datasets are available here:  /root/pipeline/datasets/...

ll /root/pipeline/datasets

# Prepare input for the format expected by the graph
t_input = tf.placeholder(np.float32, name='our_input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)

# Load graph and import into graph used by our session
model_fn = '/root/pipeline/datasets/inception/tensorflow_inception_graph.pb'
graph_def = tf.GraphDef.FromString(open(model_fn).read())
tf.import_graph_def(graph_def, {'input':t_preprocessed})

from IPython.display import clear_output, Image, display, HTML

# Helper functions for TF Graph visualization
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
#show_graph(tmp_def)

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')  # Blurred image -- low frequencies only
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1]) 
        hi = img-lo2 # hi is img with low frequencies removed
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in xrange(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1] # List of images with lower and lower frequencies

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img # Reconstructed image, all frequencies added back together

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n) # Split into frequencies
    tlevels = map(normalize_std, tlevels) # Normalize each frequency band
    out = lap_merge(tlevels) # Put image back together
    return out[0,:,:,:]

# Showing the lap_normalize graph with TensorBoard
lap_graph = tf.Graph()
with lap_graph.as_default():
    lap_in = tf.placeholder(np.float32, name='lap_in')
    lap_out = lap_normalize(lap_in)
show_graph(lap_graph)





