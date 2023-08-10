from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from process_data import process_data
from wrap_functions import doublewrap
from wrap_functions import define_scope

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64 # Number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2000

class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        # config
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        
        # ops
        with tf.device('/gpu:0'):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.create_placeholders()
            
            self.prediction
            self.loss
            self.optimize
            self.summaries
        
    def create_placeholders(self):
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')
          
    
    @define_scope(scope='embed')
    def prediction(self):
        self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                         self.embed_size], -1.0, 1.0),
                                                         name='embed_matrix')
        embed_predict = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed_predict')
        return embed_predict
    
    @define_scope
    def loss(self):
        nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                    stddev=1.0 / (self.embed_size ** 0.5)),
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')
        
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                            biases=nce_bias,
                                            labels=self.target_words,
                                            inputs=self.prediction,
                                            num_sampled=self.num_sampled,
                                            num_classes=self.vocab_size), name='nce_loss')
        return loss
    
    @define_scope
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss, global_step=self.global_step)
    
    @define_scope
    def summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('histogram_loss', self.loss)
        return tf.summary.merge_all()

def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver()
    
    initial_step = 0
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        total_loss = 0.0
        writer = tf.summary.FileWriter('./improved_graph/lr' + str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in xrange(initial_step, initial_step + num_train_steps):
            centers, targets = batch_gen.next()
            feed_dict = {model.center_words: centers, model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimize, model.summaries],
                                             feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if(index + 1) % SKIP_STEP == 0:
                print('Average loss at step{}:{:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, './checkpoints/skip-gram', index)
                
        ####################
        # code to visualize the embeddings. uncomment the below to visualize embeddings
        final_embed_matrix = sess.run(model.embed_matrix)

        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'processed/vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)

def main():

    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
        
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == '__main__':
    main()

