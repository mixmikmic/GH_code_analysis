import numpy as np
import time 
import tensorflow as tf
import scipy.misc
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Downloading the model, if it does not exist
import urllib
import os
if not os.path.isfile('texture_net_frozen.pb'):
    urllib.urlretrieve(
    "https://dl.dropboxusercontent.com/u/9154523/models/texture_net/texture_net_frozen.pb",
    "texture_net_frozen.pb")
get_ipython().magic('ls -hl texture_net_frozen.pb')

content_image = scipy.misc.imread('poodle.jpg')
plt.imshow(content_image)
content_image = content_image.astype(np.float)

with tf.gfile.GFile('texture_net_frozen.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

tf.reset_default_graph()
graph = tf.Graph().as_default() 
tf.import_graph_def(graph_def,  name='')

# Finding the correct feeds and fetches
ops = tf.get_default_graph().get_operations()
#for i in ops[0:5]:
for i in ops[0:3]:
    print(i.name)
print('...')
for i in ops[-3:]:
    print(i.name)

graph = tf.get_default_graph()
image = graph.get_tensor_by_name('image-placeholder:0')
fetch = graph.get_tensor_by_name('deprocessing/concat:0')

with tf.Session() as sess:
    t = time.time()
    res = sess.run(fetch, feed_dict={image:content_image})
    print("Transfer of {} pixels, done in {} sec".format(content_image.shape, time.time()-t))

img = np.clip(res, 0, 255).astype(np.uint8)
#plt.figure(figsize=(126,8.0))
plt.figure()
plt.imshow(img)



