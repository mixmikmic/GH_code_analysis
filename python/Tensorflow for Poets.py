import os
import argparse
import urllib.request
import tensorflow as tf
from IPython.display import Image

import retrain  # the Tensorflow script retrain.py we downloaded

# default values
# For the meaning of these values look at retrain.py:
FLAGS = argparse.Namespace()

FLAGS.image_dir = ""
FLAGS.output_graph = '.\\retrained_graph.pb'
FLAGS.output_labels = '.\\output_labels.txt'
FLAGS.summaries_dir = '.\\summaries'
FLAGS.how_many_training_steps = 4000
FLAGS.learning_rate = 0.01
FLAGS.testing_percentage = 10
FLAGS.validation_percentage = 10
FLAGS.eval_step_interval = 10
FLAGS.train_batch_size = 100
FLAGS.test_batch_size = -1
FLAGS.validation_batch_size = 100
FLAGS.print_misclassified_test_images = False
FLAGS.model_dir = "."
FLAGS.bottleneck_dir = "bottlenecks"
FLAGS.final_tensor_name = "final_result"
FLAGS.flip_left_right = False
FLAGS.random_crop = 0
FLAGS.random_scale = 0
FLAGS.random_brightness = 0

# change default: 
FLAGS.how_many_training_steps = 500
FLAGS.model_dir = "inception"
# FLAGS.summaries_dir = "C:\\to\\temp" 
FLAGS.output_graph = "retrained_graph_v2.pb"  
FLAGS.output_labels = "retrained_labels.txt"
FLAGS.image_dir = "flower_photos"

retrain.FLAGS = FLAGS
tf.app.run(main=retrain.main)  # this is basically same as retrain.main("")

# Read in the image_data

# test_image_path = ".\\test_flowers\\dandelion.jpg"  # uncomment this if you want to use a local file
# test_image_path = "https://upload.wikimedia.org/wikipedia/commons/4/44/Tulip_-_floriade_canberra.jpg"
test_image_path = "https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/3021186b83bc90c2.png"

if test_image_path[:4] == "http":  # assuming URL
    image_data = urllib.request.urlopen(test_image_path).read()
else:  # assuming file path
    image_data = tf.gfile.FastGFile(test_image_path, 'rb').read() 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(FLAGS.output_labels)]

# Unpersists graph from file
with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor,              {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

# Output image in Jupyter
Image(url=test_image_path, width=100, height=100)



