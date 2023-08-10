import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from wrap_functions import define_scope
import functools

class Model:
    def __init__(self, input_data, label, input_dim=784, output_dim=10, learning_rate=0.01):
        # config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # ops
        self.input_data = input_data
        self.label = label
        self.prediction
        self.optimize
        self.error
        
    @define_scope
    def prediction(self):
        xvaier_stddev = tf.sqrt(1.0 / tf.cast(self.input_dim + self.output_dim, dtype=tf.float32))
        w = tf.Variable(tf.random_normal(shape=[self.input_dim, self.output_dim],
                                         stddev=xvaier_stddev), name='weights')
        b = tf.Variable(tf.zeros(shape=[self.output_dim], name='bias'))
        logits = tf.matmul(self.input_data, w) + b
        preds = tf.nn.softmax(logits)
        return preds # preds will be set as an attribute named _cache_prediction
                        
    @define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.minimize(cross_entropy)
    
    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

lr = 0.01
batch_size = 128
n_epochs = 25
n_batches = int(mnist.train.num_examples/batch_size)

def main():
    g = tf.Graph()
    with g.as_default():
        image = tf.placeholder(tf.float32, [None, 784])
        label = tf.placeholder(tf.float32, [None, 10])
        model = Model(image, label)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_epochs):
          images, labels = mnist.test.images, mnist.test.labels
          error = sess.run(model.error, {image: images, label: labels})
          print('Epoch {} : Test error {:6.2f}%'.format(i, 100 * error))
          for _ in range(60):
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(model.optimize, {image: images, label: labels})

        writer = tf.summary.FileWriter('./my_graph/03/logstic_reg', sess.graph)
        writer.close()
        
if __name__ == '__main__':
  main()

