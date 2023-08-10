import tensorflow as tf
from util import get_log_path

my_graph = tf.Graph()
with my_graph.as_default():
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b

print(a)
print(b)
print(c)

print(a.graph)
print(b.graph)
print(c.graph)
old = c.graph

my_graph.as_graph_def()

with tf.Session(graph=my_graph) as sess:
    a = a.eval()
    b = sess.run(b)
    c = sess.run(c)    
    print(a)
    print(b)
    print(c)

sess = tf.InteractiveSession()
a = tf.constant([[2, 2], [-8.3, 20]], name="a")
b = tf.constant([[1, 2, 4], [-1.1, 2.2, 4.3]], name="b")
c = tf.matmul(a, b, name="c")
d = tf.constant([1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 7, 8, 7, 656, 4])
e = tf.reduce_mean(d)
f = tf.reduce_max(d)
g = tf.reduce_sum(d)
print(a)
print(b)
print(c)

new = c.graph
print("new = ", new)
print("old = ", old)

print("matrix multiplication  = {}\n".format(c.eval()))
print("mean  = ", e.eval())
print("max  = ", f.eval())
print("sum  = ", g.eval())
sess.close()

my_graph = tf.Graph()
with my_graph.as_default():
    a = tf.constant(5.0, name="a")
    b = tf.constant(6.0, name="b")
    c = tf.multiply(a, b, name="c")
with tf.Session(graph=my_graph) as sess:
    log_path = get_log_path()
    print("\n&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&")
    print("\ntensorboard  --logdir={}\n".format(log_path))
    writer = tf.summary.FileWriter(log_path, sess.graph)
    print(sess.run(a))

# Type the command above for tensorboard visualization

graph = tf.Graph() 
with graph.as_default():
    zeros1 = tf.zeros([2, 3], tf.int32, name="zeros1") 
    input_tensor = tf.constant([[0, 1],[0, 1]], name="input")
    zeros2 = tf.zeros_like(input_tensor, name="zeros2")
    ones1 = tf.ones([2, 3], tf.int32, name="ones1") 
    ones2 = tf.ones_like(input_tensor, name="ones2")
    only69 = tf.fill([2, 3], 69,name="only69") 
with tf.Session(graph=graph) as sess:
    x1, x2, x3, x4, x5 = sess.run([zeros1,
                                   zeros2,
                                   ones1,
                                   ones2,
                                   only69])
    print(x1,"\n")
    print(x2,"\n")
    print(x3,"\n")
    print(x4,"\n")
    print(x5,"\n")            

graph = tf.Graph() 
with graph.as_default():
    a = tf.Variable(2, name="scalar")
    b = tf.Variable([2, 3], name="vector")
    c = tf.Variable([[0, 1], [2, 3]], name="matrix")
    W = tf.Variable(tf.zeros([784, 10]))
    init = tf.global_variables_initializer()
with tf.Session(graph=graph) as sess:
    sess.run(init)
    x1, x2, x3, x4 = sess.run([a, b, c, W])
    print(x1,"\n")
    print(x2,"\n")
    print(x3,"\n")
    print(x4,"\n")

graph = tf.Graph() 
with graph.as_default():
    W = tf.Variable(tf.truncated_normal([2, 2]))
    x = tf.constant([[0, 1], [-1, 0]], dtype="float32")
    result = tf.matmul(W,x)
with tf.Session(graph=graph) as sess:
    sess.run(W.initializer)
    result = sess.run(result)
    print(result)

graph = tf.Graph() 
with graph.as_default():
    W = tf.Variable(10.0)
    assign_op = W.assign(2 * W)
with tf.Session(graph=graph) as sess:
    sess.run(W.initializer)
    print("original W = ", W.eval())
    sess.run(assign_op)
    print("first assign: W = ", W.eval())
    sess.run(assign_op)
    print("second assign: W = ", W.eval())
    sess.run(W.assign_add(100.)) 
    print("third assign: W = ", W.eval())
    sess.run(W.assign_sub(200.)) 
    print("fourth assign: W = ", W.eval())
with tf.Session(graph=graph) as sess:
    sess.run(W.initializer)
    print("original W = ", W.eval())

graph = tf.Graph() 
with graph.as_default():
    W = tf.Variable(10.)
print("First Session")
with tf.Session(graph=graph) as sess:
    sess.run(W.initializer)
    print("original W = ", W.eval())
    sess.run(W.assign_add(2.)) 
    print("first assign: W = ", W.eval())
print("\nSecond Session")
with tf.Session(graph=graph) as sess:
    sess.run(W.initializer)
    print("original W = ", W.eval())
    sess.run(W.assign_sub(2.)) 
    print("second assign: W = ", W.eval())

graph = tf.Graph() 
with graph.as_default():
    a = tf.placeholder(tf.float32, shape=[3])
    b = tf.constant([5, 5, 5], tf.float32)
    c = a + b 
with tf.Session(graph=graph) as sess:
    all_dict = {0:{a: [1, 2, 3]}, 1:{a: [1, 2, 3], b:[1, 2, 3]}}
    for i in [0,1]:
        feed_dict = all_dict[i]
        result = sess.run(c,feed_dict=feed_dict)
        print(result)

graph1 = tf.Graph() 
with graph1.as_default():
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = tf.add(a, b) # the node is created before executing the graph
with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        print(sess.run(c))

graph2 = tf.Graph() 
with graph2.as_default():
    a = tf.constant(5.0)
    b = tf.constant(6.0)
with tf.Session(graph=graph2) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        print(sess.run(tf.add(a, b)))  # the node is created while executing the graph

graph1.as_graph_def()

graph2.as_graph_def()



