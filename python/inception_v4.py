import tensorflow as tf
import os
import inception
import inception_utils

slim = tf.contrib.slim

scope = inception.inception_v4_arg_scope()

inputs = tf.placeholder(tf.float32, (None, 299, 299, 3), "input")
        
with slim.arg_scope(scope):
    print(inputs)
    net, end_points = inception.inception_v4(inputs, is_training=False)

saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.

inception_session = tf.Session()
inception_session.run(tf.global_variables_initializer())

# Restore variables from disk.
saver.restore(inception_session, "checkpoints/inception_v4.ckpt")
print("Model restored.")

image_path = os.path.join('images/', 'canoe.jpg')

with tf.variable_scope('image'):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    #we want to use decode_image here but it's buggy
    decoded = tf.image.decode_jpeg(image_data, channels=None)
    normed = tf.divide(tf.cast(decoded, tf.float32), 255.0)
    batched = tf.expand_dims(normed, 0)
    resized_image = tf.image.resize_bilinear(batched, [299, 299])
    standard_size = resized_image
    graph_norm = standard_size * 255.0
    
with tf.Session() as sess:
    raw_image, file_image, plot_image = sess.run((decoded, graph_norm, standard_size), feed_dict={})

#This is the normalization the network expects
feed_image = (file_image - 128) / 128

print(feed_image.shape)
print(file_image.shape)

predictions = inception_session.run((net), feed_dict={'input:0': feed_image})

print(predictions)

from tensorflow.python.framework import graph_util
from tensorflow.python.training import saver as saver_lib
from tensorflow.core.protobuf import saver_pb2

checkpoint_prefix = os.path.join("checkpoints", "saved_checkpoint")
checkpoint_state_name = "checkpoint_state"

input_graph_name = "inception_v4_prefreeze.pb"
output_graph_name = "inception_v4.pb"

input_graph_path = os.path.join("checkpoints", input_graph_name)

saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

checkpoint_path = saver.save(
  inception_session,
  checkpoint_prefix,
  global_step=0,
  latest_filename=checkpoint_state_name)

graph_def = inception_session.graph.as_graph_def()

from tensorflow.python.lib.io import file_io

file_io.atomic_write_string_to_file(input_graph_path, str(graph_def))
print("wroteIt")

train_writer = tf.summary.FileWriter('summaries/' + 'graphs/inception',
                                      inception_session.graph)

from tensorflow.python.tools import freeze_graph

input_saver_def_path = ""
input_binary = False
output_node_names = "InceptionV4/Logits/Predictions"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"

output_graph_path = os.path.join("data", output_graph_name)
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

