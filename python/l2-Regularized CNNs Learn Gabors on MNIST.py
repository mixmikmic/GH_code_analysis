get_ipython().magic('env CUDA_VISIBLE_DEVICES=1')

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import utils.CNN as CNN

get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train(graph,num_minibatches,batch_size=64,showEvery=1000):
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        x = ops['inputs']; y_ = ops['targets']
        accuracy = ops['accuracy']
        train_step = ops['train_step']
        keep_prob = ops['keep_prob']
        filters = ops['filters']

        for i in range(num_minibatches):
            batch = mnist.train.next_batch(batch_size)
            if i % showEvery == 0:
                train_accuracy = accuracy.eval(feed_dict={
                                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
        weights = filters.eval()
    
    return weights

regularized_graph, ops = CNN.specifyGraph(weight_decay=2e-1)

regularized_weights = train(regularized_graph,10000)

unregularized_graph, ops = CNN.specifyGraph(weight_decay=0.)

unregularized_weights = train(unregularized_graph,10000)

CNN.plotFilters(unregularized_weights, title="Unregularized")

CNN.plotFilters(regularized_weights,title=r"$\ell^2$ Regularized")

gaussian_random_weights = np.random.standard_normal(size=regularized_weights.shape)

randomWeightIdx = np.random.choice(unregularized_weights.shape[-1])
no_reg_weight = unregularized_weights[:,:,0,randomWeightIdx]
reg_weight = regularized_weights[:,:,0,randomWeightIdx]
randomized_weight = gaussian_random_weights[:,:,0,randomWeightIdx]

weights_to_show = [no_reg_weight, reg_weight, randomized_weight]
names = ["No Regularization", r"$\ell^2$ Regularization", "Gaussian Random"]

CNN.plotAutoCorrs(weights_to_show,names)

