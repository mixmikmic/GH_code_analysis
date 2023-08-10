# boilerplate code
import os
import re
from cStringIO import StringIO
import numpy as np
from functools import partial
import PIL.Image
import PIL.ImageOps
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf

print "Done"

# creating fresh Graph and TensorFlow session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# Prepare input for the format expected by the graph
t_input = tf.placeholder(np.float32, name='our_input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)

# Load graph and import into graph used by our session
model_fn = 'imagenet/tensorflow_inception_graph.pb'
graph_def = tf.GraphDef.FromString(open(model_fn).read())
tf.import_graph_def(graph_def, {'input':t_preprocessed})

print "imported"

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

print 'Number of layers', len(layers)
print 'Total number of feature channels:', sum(feature_nums)

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
show_graph(tmp_def)

label_lookup_path = 'imagenet/imagenet_2012_challenge_label_map_proto.pbtxt'
uid_lookup_path = 'imagenet/imagenet_synset_to_human_label_map.txt'

# The id translation goes via id strings -- find translation between UID string and human-friendly names
proto_as_ascii_lines = open(uid_lookup_path).readlines()
uid_to_human = {}
p = re.compile(r'[n\d]*[ \S,]*')
for line in proto_as_ascii_lines:
    parsed_items = p.findall(line)
    uid = parsed_items[0]
    human_string = parsed_items[2]
    uid_to_human[uid] = human_string

# Get node IDs to UID strings map
proto_as_ascii_lines = open(label_lookup_path).readlines()
node_id_to_uid = {}
for line in proto_as_ascii_lines:
    if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

# Make node ID to human friendly names map
node_id_to_name = {}
for key, val in node_id_to_uid.iteritems():
    name = uid_to_human[val]
    node_id_to_name[key] = name
    
# make sure we have a name for each possible ID
for i in range(graph.get_tensor_by_name('import/softmax2:0').get_shape()[1]):
    if i not in node_id_to_name:
        node_id_to_name[i] = '???'

node_id_to_name[438]

# Helper function to get a named layer from the graph
def T(layer_name):
    return graph.get_tensor_by_name("import/%s:0" % layer_name)

softmax = T('softmax2')

def prep_img(filename):
    size = (224, 224)
    img = PIL.Image.open(filename)
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    thumb = PIL.ImageOps.fit(img, size, PIL.Image.ANTIALIAS, (0.5, 0.5))
    return np.float32(thumb)

# Print the 5 top predictions for a given image
def prediction(filename, k=5):
    img = prep_img(filename)
    # Load, resize, and central square crop the image.
    
    # Compute predictions
    predictions = sess.run(softmax, {t_input: img})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-k:][::-1]
    for node_id in top_k:
        human_string = node_id_to_name[node_id]
        score = predictions[node_id]
        print '%s (score = %.5f)' % (human_string, score)
        
# Helper function: Display an image
def showarray(a, fmt='jpeg', size=None):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = StringIO()
    img = PIL.Image.fromarray(a)
    if size is not None:
        img = img.resize((size,size))
    img.save(f, fmt)
    display(Image(data=f.getvalue()))
        
print "defined"

showarray(prep_img('testimages/deer.jpg')/255., size=500)
prediction('testimages/deer.jpg')

# Normalize an image for visualization
def visstd(a, s=0.1):
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

img_noise = np.random.uniform(size=(224,224,3)) + 100.0

# Uncomment the following line to see all the possible weights
print '\n'.join([op.name[7:] for op in graph.get_operations() if op.name.endswith('w')])
w = T('conv2d0_w')
shape = w.get_shape().as_list()
print shape
feature_id = 14
weights = tf.squeeze(tf.slice(w, [0, 0, 0, feature_id], [-1, -1, -1, 1])).eval()
showarray(visstd(weights), size=200)

a = T('conv2d0')
feature = tf.squeeze(tf.slice(a, [0, 0, 0, feature_id], [-1, -1, -1, 1]))
img = prep_img('testimages/deer.jpg')
feature = sess.run(feature, {t_input: img})
showarray(visstd(feature), size=500)

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for i in xrange(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        print score,
    clear_output()
    showarray(visstd(img), size=448)

# Run this cell if you'd like a list of all layers to pick from
print '\n'.join([op.name[7:] for op in graph.get_operations() if op.name.endswith('pre_relu')])

# Pick any internal layer. 
layer = 'mixed4d_3x3_bottleneck_pre_relu'
print '%d channels in layer.' % T(layer).get_shape()[-1] 

# Pick a feature channel to visualize
channel = 139

render_naive(T(layer)[:,:,:,channel])

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = map(tf.placeholder, argtypes)
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in xrange(0, max(h-sz//2, sz),sz):
        for x in xrange(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=2, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for octave in xrange(octave_n):
        if octave > 0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in xrange(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            print '.',
        clear_output()
        showarray(visstd(img))
        print 'Octave %d' % octave

render_multiscale(T(layer)[:,:,:,channel])

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

def render_lapnorm(t_obj, img0=img_noise, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # Build the Laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in xrange(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in xrange(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)  # New!
            img += g*step
            print '.',
        clear_output()
        showarray(visfunc(img))

render_lapnorm(T(layer)[:,:,:,channel])

render_lapnorm(T(layer)[:,:,:,65])

render_lapnorm(T('mixed3b_1x1_pre_relu')[:,:,:,101])

render_lapnorm(T(layer)[:,:,:,65] + T(layer)[:,:,:,139], octave_n=4)

def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # Build the Laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=4))

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in xrange(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in xrange(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in xrange(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print '.',
        clear_output()
        showarray(img/255.0)
        
    return img

img0 = PIL.Image.open('testimages/mountains.jpg')
img0 = np.float32(img0)
showarray(img0/255.0)

prediction('testimages/mountains.jpg')

_ = render_deepdream(tf.square(T('mixed4c')), img0)

_ = render_deepdream(T(layer)[:,:,:,65], img0)

frame = img0
h, w = frame.shape[:2]
s = 0.05 # scale coefficient
for i in xrange(100):
    frame = render_deepdream(tf.square(T('mixed4c')), img0=frame)
    img = PIL.Image.fromarray(np.uint8(np.clip(frame, 0, 255)))
    img.save("dream-%04d.jpg"%i)
    # Zoom in while maintaining size
    img = img.resize(np.int32([w*(1+s), h*(1+s)]))
    t, l = np.int32([h*(1+s) * s / 2, w*(1+s) * s / 2])
    img = img.crop([l, t, w-l, h-t])
    img.load()
    print img.size
    frame = np.float32(img)

