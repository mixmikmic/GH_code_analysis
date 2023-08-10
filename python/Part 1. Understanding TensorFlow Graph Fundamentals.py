import tensorflow as tf
from src import cloud_visualizer

g1 = tf.Graph() 

with g1.as_default(): 
    a = tf.constant(-0.8, name="A") # or a = -0.8
    b = tf.abs(a, name="B")
    
    c = tf.cos(b, name="C")
    d = tf.ceil(b, name="D")
    
    e = tf.multiply(c, d, name="E")
    f = tf.acos(c, name="F")
    
    g = tf.asin(e, name="G")
    
tf.summary.FileWriter("logs", g1).close() # write graph out to tensorboard for visualization
cloud_visualizer.show_graph(g1)

from google.protobuf import text_format

# Lets export the graphdef protobuf as a readable text file.
with open("./storage/g1.pbtxt", "w") as f:
    f.write(text_format.MessageToString(g1.as_graph_def()))

g1.as_graph_def() # display the graph inline.

from google.protobuf import text_format

# Lets import our new graphdef protobuf
with open("./storage/g2.pbtxt", "r") as f: 
    graphdef = text_format.Parse(f.read(), tf.GraphDef())

g2 = tf.Graph() 
with g2.as_default():
    tf.import_graph_def(graph_def=graphdef, name="") # import the graph def into our new graph.

cloud_visualizer.show_graph(g2)

sess = tf.Session(graph=g2);
print(sess.run(g2.get_tensor_by_name("Z:0")))

import tensorflow as tf
from src import cloud_visualizer

g3 = tf.Graph() 

with g3.as_default(): 
    a = tf.placeholder(dtype=tf.float32, shape=None, name="A") # or a = -0.8
    b = tf.abs(a, name="B")
    
    c = tf.cos(b, name="C")
    d = tf.ceil(b, name="D")
    
    e = tf.multiply(c, d, name="E")
    f = tf.acos(c, name="F")
    
    g = tf.asin(e, name="G")
    
    z = tf.add(f, g, name="Z")
    
tf.summary.FileWriter("logs", g3).close() # write graph out to tensorboard for visualization
cloud_visualizer.show_graph(g3)

sess = tf.Session(graph=g3)
sess.run("Z:0", feed_dict={a: [[1, 2, -3], [2, 1, 0]]}) # Can take in arbitrary shapes of data to eval.

import tensorflow as tf
from src import cloud_visualizer

g1 = tf.Graph() 

with g1.as_default(): 
    a = -0.8
    b = tf.abs(a)
    
    c = tf.cos(b)
    d = tf.ceil(b)
    
    e = c*d
    f = tf.acos(c)
    
    g = tf.asin(e)
    
tf.summary.FileWriter("logs", g1).close() # write graph out to tensorboard for visualization
cloud_visualizer.show_graph(g1)

