#setup Jupyter for TensorBoard inline display
import numpy as np
from IPython.display import clear_output, Image, display, HTML

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
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

import tensorflow as tf

# build a linear model where y = w * x + b

w = tf.Variable([0.2], tf.float32, name='weight')
b = tf.Variable([0.3], tf.float32, name='bias')

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name='Y')

# the training values for x and y
x = ([2.,3.,4.,5.])
y = ([-1.,-2.,-3.,-4.])

# define the linear model
linear_model = w*X+b

# define the loss function
square_delta = tf.square(linear_model - Y)
loss = tf.reduce_sum(square_delta)

#set the learning rate and training epoch
learning_rate = 0.01
training_epoch = 1000

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# start a session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
        
    for i in range(training_epoch):
        sess.run(train, feed_dict={X:x,Y:y})
           
    # evaluate training accuracy
    curr_w, curr_b, curr_loss  = sess.run([w, b, loss], {X:x,Y:y})
    print('w: %f b: %f loss: %f '%(curr_w, curr_b, curr_loss))

#call to display TensorBoard
show_graph(tf.get_default_graph().as_graph_def())

