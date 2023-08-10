import tensorflow as tf
from IPython.display import clear_output, Image, display, HTML
import numpy as np
import pylab as pl
from sklearn.datasets.samples_generator import make_blobs

get_ipython().run_line_magic('matplotlib', 'inline')


# Helper functions to inline visualization of computing graphs
# Extracted from: 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
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

# Functions for plotting 2D data and decision regions

def plot_data(X, y):
    y_unique = np.unique(y)
    colors = pl.cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        pl.scatter(this_X[:, 0], this_X[:, 1],  c=color,
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y)
    pl.legend(loc="best")
    pl.title("Data")

def plot_decision_region(X, pred_fun):
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    x_vals = np.linspace(min_x, max_x, 30)
    y_vals = np.linspace(min_y, max_y, 30)
    XX, YY = np.meshgrid(x_vals, y_vals)
    grid_r, grid_c = XX.shape
    ZZ = np.zeros((grid_r, grid_c))
    for i in range(grid_r):
        for j in range(grid_c):
            ZZ[i, j] = pred_fun(XX[i, j], YY[i, j])
    pl.contourf(XX, YY, ZZ, 30, cmap = pl.cm.coolwarm, vmin= -1, vmax=2)
    pl.colorbar()
    pl.xlabel("x")
    pl.ylabel("y")

graph = tf.Graph()

with graph.as_default():
    a = tf.constant(10, tf.float32, name= 'a')
    b = tf.constant(-5, tf.float32, name= 'b')
    c = tf.constant(4, tf.float32, name= 'c')

    x = tf.placeholder(tf.float32, name= 'x')

    y = a * x * x + b * x + c
    
show_graph(graph.as_graph_def())

y

with graph.as_default():
    sess = tf.Session()
    result = sess.run(y, {x: 5.0})
    sess.close()

print('y =', result)

with graph.as_default():
    y_prime = tf.gradients(y, [x])
    
    sess = tf.Session()
    result = sess.run(y_prime, {x: 5.0})
    sess.close()

print(result)

show_graph(graph.as_graph_def())

graph = tf.Graph()
with graph.as_default():
    a = tf.constant(10, tf.float32, name= 'a')
    b = tf.constant(-5, tf.float32, name= 'b')
    c = tf.constant(4, tf.float32, name= 'c')
    x = tf.Variable(0.0, name= 'x')
    y = a * x * x + b * x + c

    optimizer = tf.train.GradientDescentOptimizer(0.02)
    update = optimizer.minimize(y)

    # Graph execution
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        val_y, val_x, _ = sess.run([y, x, update])
        print(i, val_x, val_y)
    sess.close()

show_graph(graph.as_graph_def())

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
pl.figure(figsize=(8, 6))
plot_data(X, Y)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32,shape=[None,2])
    y_true = tf.placeholder(tf.float32,shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)
  
    with tf.name_scope('train') as scope:
        learning_rate = 1.0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

show_graph(graph.as_graph_def())

num_epochs = 50
losses = []

with graph.as_default():
    sess = tf.Session()
    sess.run(init)      
    for step in range(num_epochs):
        sess.run(train,{x: X, y_true: Y})
        if (step % 5 == 0):
            losses.append(sess.run(loss, {x: X, y_true: Y}))
                       
pl.figure(figsize = (8,16/3))
pl.plot(losses)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

with graph.as_default():
    wval = sess.run(w)
    print(wval)
    result = sess.run(y_pred, {x:np.array([[1,2]])})
    print(result)
    def pred_fun(x1, x2):
        xval = np.array([[x1, x2]])
        return sigmoid(sess.run(y_pred,{x: xval}))

pl.figure(figsize = (8,16/3))    
plot_decision_region(X, pred_fun)
plot_data(X, Y)

sess.close()

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32,shape=[None,2])
    y_true = tf.placeholder(tf.float32,shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b
        variable_summaries(w)
        variable_summaries(b)

    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)
        variable_summaries(loss)
  
    with tf.name_scope('train') as scope:
        learning_rate = 1.0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

show_graph(graph.as_graph_def())

x_train = X[:50]
x_test  = X[50:]
y_train = Y[:50]
y_test  = Y[50:]

LOG_DIR = 'logs'
train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                 graph=graph)
test_writer = tf.summary.FileWriter(LOG_DIR + '/test')

num_epochs = 400

with graph.as_default():
    with tf.Session() as sess:
        sess.run(init)      
        for step in range(num_epochs):
            summary, train_loss, _ = sess.run([merged, loss, train] ,{x: x_train, y_true: y_train})
            train_writer.add_summary(summary, step)
            summary, val_loss = sess.run([merged, loss] ,{x: x_test, y_true: y_test})
            test_writer.add_summary(summary, step)
            if step % 20 == 0:
                print(step, train_loss, val_loss)
                



