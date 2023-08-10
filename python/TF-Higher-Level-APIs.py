import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    i = tf.constant(0)
    condition = lambda i: tf.less(i, 10)
    body = lambda i: tf.add(i, 1)
    r = tf.while_loop(condition, body, [i])
    print(sess.run(r))

import collections
Pair = collections.namedtuple('Pair', 'j, k')
with tf.Session() as sess:
    ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
    c = lambda i, p: i < 10
    b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
    ijk_final = tf.while_loop(c, b, ijk_0)
    print(sess.run(ijk_final))

def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)

def condition(x):
    return tf.reduce_sum(x) < 100

x = tf.Variable(tf.constant(0, shape=[2, 2]))

with tf.Session():
    tf.initialize_all_variables().run()
    result = tf.while_loop(condition, body, [x])
    print(result.eval())

elems = np.array([[1,2], [2,6], [3,6], [4,6], [5,6], [6,6]])
elems.shape

squares = tf.map_fn(lambda x: x * x, elems)

with tf.Session() as sess:
    print(sess.run(squares))
print(tf.get_default_session())    
print(squares.eval(session=tf.get_default_session()))

import numpy as np
import tensorflow as tf

a = np.random.randint(-1, 5, (3,4,2))
a


def get_sequence_length(sequence):
    '''
    Returns the sequence length, droping out all the zeros if the sequence is padded
    :param sequence: Tensor(shape=[batch_size, doc_length, feature_dim])
    :return: Array of Document lengths of size batch_size
    '''
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used,1)
    length = tf.cast(length, tf.int32)
    return length

3 * 4 * 2

#3 samples, doc_length=4+2+1, num_words = 2  ===> 3 * 7 * 2 = 42
#3 samples, doc_length=4, num_words = 2 PAdded ====> 3 * 4 * 2 = 24
a = np.array([[
        [ 2,  3],
        [ 4,  2],
        [ 1,  0],
        [ 4,  4]],

       [[ 2, 1],
        [ 2, 1],
        [ -2,  -3],
        [ -3,  -3]],

       [[1,  2],
        [ -3,  -2],
        [ -4,  -2],
        [ -2,  -3]]])
a

with tf.Session() as sess:
    char_ids = tf.Variable(a, dtype=tf.int32)
    value = tf.constant(0, tf.int32)
    
    length = get_sequence_length(char_ids)
    flag = tf.greater_equal(char_ids, 0)
    res = tf.gather_nd(indices=tf.where(flag), params=char_ids)
    res = tf.reshape(res, shape=(3, 2, -1))
    tf.initialize_all_variables()
    print(sess.run(char_ids, feed_dict={char_ids: a}))
    print(sess.run(length, feed_dict={char_ids: a}))
    print(sess.run(res, feed_dict={char_ids: a}))


x = tf.constant([1, 2, 0, 4])
y = tf.Variable([1, 2, 0, 4])
mask = a > 0
slice_y_greater_than_one = tf.boolean_mask(a, mask)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(slice_y_greater_than_one))



