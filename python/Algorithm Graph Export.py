from flowdec import restoration as fd_restoration
import tensorflow as tf
import tempfile
import shutil
import os

ndims = 3
domain_type = 'complex'
algo = fd_restoration.RichardsonLucyDeconvolver(
    ndims, pad_mode='log2', real_domain_fft=(domain_type == 'real')
).initialize()

export_dir = tempfile.mkdtemp('-graph', 'tf-')
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
export_dir

algo.graph.save(export_dir, save_as_text=True)

g = tf.Graph()
with tf.Session(graph=g) as sess:
    saver = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir) 

import numpy as np
with tf.Session(graph=g) as sess:
    dh = g.get_tensor_by_name("data:0")
    kh = g.get_tensor_by_name("kernel:0")
    ph = g.get_tensor_by_name("pad_mode:0")
    ih = g.get_tensor_by_name("niter:0")
    o = g.get_tensor_by_name("result:0")
    res = sess.run(o, feed_dict={dh: np.ones([5]*ndims), kh: np.ones([5]*ndims), ph: 'log2', ih: 10})

res.shape

