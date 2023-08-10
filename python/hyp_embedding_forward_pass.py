import tensorflow as tf

import math









def arctanh(x):
    return tf.log(tf.divide(1+x,1-x))

def inner_prod(r_in, r_out, theta_in, theta_out):
    cosine = tf.cos(theta_in - theta_out)
    radius = tf.multiply(arctanh(r_in), arctanh(r_out))
    return 4 * tf.multiply(cosine, radius)

def tensor_inner_prod(r_example, r_sample, theta_example, theta_sample):
    r1 = arctanh(r_example)
    r2 = arctanh(r_sample)
    radius_term = r1[:, None] + r2[None, :]
    cos_term = theta_example[:, None] - theta_sample[None, :]
    return tf.squeeze(4* tf.multiply(cos_term, radius_term))

def nce_loss(true_logits, sampled_logits):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / 2
        return nce_loss_tensor

radius_in = tf.Variable(tf.sqrt(tf.random_uniform([5,1])))  # radius
# radius_out = tf.Variable(tf.zeros([5,1]))
radius_out = tf.Variable(tf.sqrt(tf.random_uniform([5,1])))
theta_in = tf.Variable(2*np.pi*tf.random_uniform([5,1]))  # angle
theta_out = tf.Variable(2*np.pi*tf.random_uniform([5,1]))
# theta_out = tf.Variable(tf.zeros([5,1]))
sm_b = tf.Variable(tf.zeros([5,1]))
examples = tf.Variable([1,2])
labels = tf.Variable([3,4])

example_radius = tf.nn.embedding_lookup(radius_in, examples)
example_theta = tf.nn.embedding_lookup(theta_in, examples)
true_radius = tf.nn.embedding_lookup(radius_out, labels)
true_theta = tf.nn.embedding_lookup(theta_out, labels)

sampled_ids = tf.Variable([0,1,2])
sampled_radius = tf.nn.embedding_lookup(radius_out, sampled_ids)
sampled_theta = tf.nn.embedding_lookup(theta_out, sampled_ids)
true_b = tf.nn.embedding_lookup(sm_b, labels)

true_logits = inner_prod(example_radius, true_radius, example_theta, true_theta) + true_b

sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
sampled_b_vec = tf.reshape(sampled_b, sampled_ids.get_shape())

sampled_logits = tensor_inner_prod(example_radius, sampled_radius, example_theta, sampled_theta) + sampled_b_vec

r1 = arctanh(example_radius)
r2 = arctanh(sampled_radius)
radius_term = r1[:, None] + r2[None, :]
cos_term = example_theta[:, None] - sampled_theta[None, :]
retval = 4* tf.multiply(cos_term, radius_term)

loss = nce_loss(true_logits, sampled_logits)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(sampled_radius))
# print(sess.run(r1))
# print(sess.run(r2))
# print(sess.run(radius_term))
# print(sess.run(cos_term))
# print(sess.run(retval))
# print(sess.run(sampled_logits))
print(sess.run(true_logits))
print(sess.run(loss))

r1.get_shape()

tf.squeeze(example_radius).get_shape()

retval.get_shape()

sampled_logits.get_shape()

x1 = np.array([1,2])
x2 = np.array([3,4])
x3 = np.concatenate((x1.T,x2.T), axis=1)
x3



