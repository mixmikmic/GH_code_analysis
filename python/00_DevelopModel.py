import numpy as np
import tensorflow as tf
from IPython.display import Image as show_image
from PIL import Image
from tensorflow.contrib.slim.nets import resnet_v1

get_ipython().system('wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz')

get_ipython().system('tar xvf resnet_v1_152_2016_08_28.tar.gz')

get_ipython().system('wget "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"')

# Placeholders
input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')

# Load the model
sess = tf.Session()
arg_scope = resnet_v1.resnet_arg_scope()
with tf.contrib.slim.arg_scope(arg_scope):
    logits, _ = resnet_v1.resnet_v1_152(input_tensor, num_classes=1000, is_training=False)

probabilities = tf.nn.softmax(logits)

checkpoint_file = 'resnet_v1_152.ckpt'
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)

get_ipython().system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg')

show_image("220px-Lynx_lynx_poing.jpg")

im = Image.open("220px-Lynx_lynx_poing.jpg").resize((224,224))
im = np.array(im)
im = np.expand_dims(im, 0)

pred, pred_proba = sess.run([logits,probabilities], feed_dict={input_tensor: im})

def create_label_lookup():
    with open('synset.txt', 'r') as f:
        label_list = [l.rstrip() for l in f]
    def _label_lookup(*label_locks):
        return [label_list[l] for l in label_locks]
    return _label_lookup

label_lookup = create_label_lookup()

top_results = np.flip(np.sort(pred_proba.squeeze()), 0)[:3]

labels=label_lookup(*np.flip(np.argsort(pred_proba.squeeze()), 0)[:3])

dict(zip(labels, top_results))

