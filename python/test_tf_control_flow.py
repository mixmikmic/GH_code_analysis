import numpy as np
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    # graph: variable b increment by X_in
    X_in = tf.placeholder(dtype=tf.float32, shape=5)
    b = tf.Variable(dtype=tf.float32, initial_value=np.zeros(5))
    assign_op = tf.assign(b, b + X_in)     # assign operation, has to be run to update b


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    feed_dict = {X_in: np.ones(5)}
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b only,   b={}'.format(b_eval))
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b only,   b={}'.format(b_eval))
    _, b_eval = session.run([assign_op, b], feed_dict=feed_dict)
    print('run assign_op and b,   b={}'.format(b_eval))
    _, b_eval = session.run([assign_op, b], feed_dict=feed_dict)
    print('run assign_op and b,   b={}'.format(b_eval))
    b_eval, _ = session.run([b, assign_op], feed_dict=feed_dict)
    print('run b and assign_op,   b={}'.format(b_eval))
    b_eval, _ = session.run([b, assign_op], feed_dict=feed_dict)
    print('run b and assign_op,   b={}'.format(b_eval))
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b only,   b={}'.format(b_eval))
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b only,   b={}'.format(b_eval))

graph = tf.Graph()
with graph.as_default():
    # graph: variable b increment by X_in
    X_in = tf.placeholder(dtype=tf.float32, shape=5)
    b = tf.Variable(dtype=tf.float32, initial_value=np.zeros(5))
    assign_op = tf.assign(b, b + X_in)     # assign operation, has to be run to update b
    with tf.control_dependencies([assign_op]):
        b_new = b*1   # this works because b_new is a operator
        # b_new = b   # this will not work because b_new is a another name of b    


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    feed_dict = {X_in: np.ones(5)}
    b_eval = session.run(b_new, feed_dict=feed_dict)
    print('run b_new,   b={}'.format(b_eval) )
    b_eval = session.run(b_new, feed_dict=feed_dict)
    print('run b_new,   b={}'.format(b_eval) )
    b_eval = session.run(b_new, feed_dict=feed_dict)
    print('run b_new,   b={}'.format(b_eval) )
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b    ,   b={}'.format(b_eval) )
    b_eval = session.run(b, feed_dict=feed_dict)
    print('run b    ,   b={}'.format(b_eval) )



