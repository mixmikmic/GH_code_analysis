import tensorflow as tf
import math

def atan2(x, y, epsilon=1.0e-12):
        """
        A hack until the tf developers implement a function that can find the angle from an x and y co-ordinate.
        :param x: 
        :param epsilon: 
        :return: 
        """
        # Add a small number to all zeros, to avoid division by zero:
        x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
        y = tf.where(tf.equal(y, 0.0), y+epsilon, y)
    
        angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
        angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
        angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
        angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
        return angle

def inner_prod(r_in, r_out, theta_in, theta_out):
        """
        Takes the hyperbolic inner product
        :param r_in: radius in the input embedding
        :param r_out: radius in the output embedding
        :param theta_in:
        :param theta_out:
        :return:
        """
        cosine = tf.cos(theta_in - theta_out)
        radius = tf.multiply(r_in, r_out)
        return tf.multiply(cosine, radius)

def tensor_inner_prod(r_example, r_sample, theta_example, theta_sample):
        """
        Calculate the inner product between the examples and the negative samples
        :param r_example:
        :param r_sample:
        :param theta_example:
        :param theta_sample:
        :return:
        """
        radius_term = tf.multiply(r_example[:, None], r_sample[None, :])
        cos_term = theta_example[:, None] - theta_sample[None, :]
        return tf.squeeze(tf.multiply(cos_term, radius_term))

vocab_size = 5
init_width = 0.1
x = tf.Variable(tf.random_uniform([vocab_size], -init_width, init_width), name="x")
y = tf.Variable(tf.random_uniform([vocab_size], -init_width, init_width), name="y")
radius_in = tf.sqrt(tf.square(x) + tf.square(y))
theta_in = atan2(x, y)

sm_x = tf.Variable(tf.zeros([vocab_size]), name="sm_x")
sm_y = tf.Variable(tf.zeros([vocab_size]), name="sm_y")
radius_out = tf.sqrt(tf.square(sm_x) + tf.square(sm_y))
theta_out = atan2(sm_x, sm_y)
sm_b = tf.Variable(tf.zeros([vocab_size]), name="sm_b")
radius_out = tf.sqrt(tf.square(sm_x) + tf.square(sm_y))
theta_out = atan2(sm_x, sm_y)

examples = tf.Variable([1,2])
labels = tf.Variable([3,4])
batch_size = tf.shape(examples)[0]
labels_matrix = tf.reshape(
                tf.cast(labels,
                        dtype=tf.int64),
                [batch_size, 1])

example_radius = tf.nn.embedding_lookup(radius_in, examples)
example_theta = tf.nn.embedding_lookup(theta_in, examples)
example_radius_hist = tf.summary.histogram('input_radius_embeddings', example_radius)
example_theta_hist = tf.summary.histogram('input_theta_embeddings', example_theta)
# Weights for labels: [batch_size, emb_dim]
true_radius = tf.nn.embedding_lookup(radius_out, labels)
true_theta = tf.nn.embedding_lookup(theta_out, labels)
true_b = tf.nn.embedding_lookup(sm_b, labels)

true_logits = inner_prod(example_radius, true_radius, example_theta, true_theta) + true_b

num_samples = 2
sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=num_samples,
                unique=True,
                range_max=5,
                distortion=0.75,
                unigrams=[1.0,1.0,1.0,1.0,1.0]))
sampled_radius = tf.nn.embedding_lookup(radius_out, sampled_ids)
sampled_theta = tf.nn.embedding_lookup(theta_out, sampled_ids)

sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)
sampled_b_vec = tf.reshape(sampled_b, [num_samples])
sampled_logits = tensor_inner_prod(example_radius, sampled_radius, example_theta,
                                                    sampled_theta) + sampled_b_vec

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(sampled_b))
# print(sess.run(sampled_w))
# print(sess.run(sampled_b_vec))
print(sess.run(true_logits))
print(sess.run(sampled_logits))



