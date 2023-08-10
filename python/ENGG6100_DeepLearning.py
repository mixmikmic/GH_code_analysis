# Import standard Python libraries
import os # for making os system calls, eg interfacing with unix 'ls' or 'mkdir'
import time # useful for approximate benchmarking
from tqdm import tqdm # progress bard
import numpy as np # numerical library for efficiently manipulating vectors and matrices outside of TensorFlow

# Differentiable programming, Deep Learning frameworks/libraries
import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import cifar10

class ConvNet:
    
    def __init__(self, x, nb_filters, use_batch_norm, phase, reuse):
        self.__dict__.update(locals())
        self.convolutional_layers(use_batch_norm, phase)
        # calculate the number of parameters to be learned in our model
        self.nb_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print("Created instance of CNN model with %d parameters" % self.nb_params)
    
    def initialize_kernel(self, shape, stddev=0.1):
        init = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=(0, 1, 2)))
        return tf.Variable(init)
    
    def get_kernel(self, shape, stddev=0.1):
        init = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=(0, 1, 2)))
        return tf.get_variable("k", initializer=init)
    
    def initialize_weight(self, shape, stddev=0.1):
        init = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0, keep_dims=True))
        return tf.Variable(init)
    
    def get_weight(self, shape, stddev=0.1):
        init = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0, keep_dims=True))
        return tf.get_variable("w", initializer=init)
    
    def bias_variable(self, shape):
        init = tf.constant(0, shape=shape)
        return tf.Variable(init)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    
    def convolutional_layers(self, use_batch_norm, phase):

        pad_k = tf.constant([[1, 1], [1, 1], [0, 0]])
        pad_a = tf.constant([[0, 0], [1, 1], [1, 1]])

        with tf.name_scope('conv_1') as scope:

            k_sz = 8 # the square kernel size (e.g 8x8)
            
            #W_conv1 = self.initialize_kernel([k_sz, k_sz, int(self.x.shape[3]), self.nb_filters])
            '''
            Try un-commmenting the above line in all layers and observing the number
            of parameters change everytime you run the below cell titled
            "Instantiate the CNN model". If we don't explicitly tell TensorFlow to re-use 
            variables, additional copies of the graph will be created, blowing up the 
            number of parameters. Reusing variables is accomplished by creating them under
            a tf.variable_scope, and using tf.get_variable rather than tf.Variable.
            If you want to force a new variable to be created everytime, you can
            set reuse=False.
            '''
            with tf.variable_scope('conv_1_init', reuse=self.reuse):
                W_conv1 = self.get_kernel([k_sz, k_sz, int(self.x.shape[3]), self.nb_filters])
                
            '''
            for logging to Tensorboard "Images" tab. 
            This is a bit of a hack to combine all 8x8 convolution
            filters into one image.
            '''
            W1_c = tf.split(W_conv1, self.nb_filters, 3)  # nb_filters x [8, 8, 3, 1]
            for i in range(self.nb_filters):
                W1_c[i] = tf.pad(tf.reshape(
                    W1_c[i], [k_sz, k_sz, int(self.x.shape[3])]), pad_k, "CONSTANT")
            W1_row0 = tf.concat(W1_c[0:8], 0)      # [80, 10, 3, 1]
            W1_row1 = tf.concat(W1_c[8:16], 0)     # [80, 10, 3, 1]
            W1_row2 = tf.concat(W1_c[16:24], 0)    # [80, 10, 3, 1]
            W1_row3 = tf.concat(W1_c[24:32], 0)    # [80, 10, 3, 1]
            W1_row4 = tf.concat(W1_c[32:40], 0)    # [80, 10, 3, 1]
            W1_row5 = tf.concat(W1_c[40:48], 0)    # [80, 10, 3, 1]
            W1_row6 = tf.concat(W1_c[48:56], 0)    # [80, 10, 3, 1]
            W1_row7 = tf.concat(W1_c[56:64], 0)    # [80, 10, 3, 1]
            W1_d = tf.concat([W1_row0, W1_row1, W1_row2, W1_row3, W1_row4,
                              W1_row5, W1_row6, W1_row7], 1)  # [80, 80, 3, 1]
            W1_e = tf.reshape(W1_d, [1, 80, 80, int(self.x.shape[3])])
            
            # Create the actual summary to appear in Tensorboard
            tf.summary.image("k1", W1_e, 1)

            # Create tensors for L1 and L2 weight decay
            self.W_conv1_p = tf.nn.l2_loss(W_conv1)
            self.W_conv1_l1 = tf.reduce_mean(tf.abs(W_conv1))
            
            h_conv1 = tf.nn.relu(tf.nn.conv2d(
                self.x, W_conv1, strides=[1, 2, 2, 1], padding='SAME'))

            a1_u, a1_var = tf.nn.moments(
                tf.abs(h_conv1), axes=[0], keep_dims=False)

            tf.summary.histogram(name='h_conv1_summ', values=h_conv1)
            tf.summary.histogram(name='W_conv1_summ', values=W_conv1)
            tf.summary.scalar("activation_mean", tf.reduce_mean(a1_u))
            tf.summary.scalar("activation_variance", tf.reduce_mean(a1_var))

        with tf.name_scope('conv_2') as scope:

            k_sz = 6 # the square kernel size
            in_ch = self.nb_filters
            out_ch = self.nb_filters * 2
            
            #self.W_conv2 = self.initialize_kernel([k_sz, k_sz, in_ch, out_ch])
            with tf.variable_scope('conv_2_init', reuse=self.reuse):
                self.W_conv2 = self.get_kernel([k_sz, k_sz, in_ch, out_ch])
            
            # Create tensors for L1 and L2 weight decay
            self.W_conv2_p = tf.nn.l2_loss(self.W_conv2)
            self.W_conv2_l1 = tf.reduce_mean(tf.abs(self.W_conv2))

            if self.use_batch_norm:
                # see https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
                # for full list of default settings.
                with tf.variable_scope('conv_2_bn', reuse=self.reuse):
                    h_conv1 = tf.contrib.layers.batch_norm(h_conv1, is_training=phase)

            W2_c = tf.split(self.W_conv2, self.nb_filters * 2,
                            3)  # f_out x [6, 6, f_in, 1]
            for i in range(self.nb_filters):
                W2_c[i] = tf.pad(tf.reshape(
                    W2_c[i], [k_sz, k_sz, self.nb_filters]), pad_k, "CONSTANT")
            W2_row0 = tf.concat(W2_c[0:8], 0)      # [64, 8, f_in, 1]
            W2_row1 = tf.concat(W2_c[8:16], 0)     # [64, 8, f_in, 1]
            W2_row2 = tf.concat(W2_c[16:24], 0)    # [64, 8, f_in, 1]
            W2_row3 = tf.concat(W2_c[24:32], 0)    # [64, 8, f_in, 1]
            W2_row4 = tf.concat(W2_c[32:40], 0)    # [64, 8, f_in, 1]
            W2_row5 = tf.concat(W2_c[40:48], 0)    # [64, 8, f_in, 1]
            W2_row6 = tf.concat(W2_c[48:56], 0)    # [64, 8, f_in, 1]
            W2_row7 = tf.concat(W2_c[56:64], 0)    # [64, 8, f_in, 1]
            W2_d = tf.concat([W2_row0, W2_row1, W2_row2, W2_row3, W2_row4,
                              W2_row5, W2_row6, W2_row7], 1)  # [64, 64, 3, 1]
            W2_e = tf.reshape(W2_d, [1, 64, 64, self.nb_filters])
            W2_f = tf.split(W2_e, self.nb_filters, 3)  # 64 x [1, 64, 64, 1]
            W2_g = tf.concat(W2_f[0:self.nb_filters], 0)
            
            # Create the summary to appear in Tensorboard
            tf.summary.image("k2", W2_g, 4)

            h_conv2 = tf.nn.relu(tf.nn.conv2d(
                h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID'))

            a2_u, a2_var = tf.nn.moments(
                tf.abs(h_conv2), axes=[0], keep_dims=False)

            tf.summary.histogram(name='W_conv2_summ', values=self.W_conv2)
            tf.summary.histogram(name='h_conv2_summ', values=h_conv2)
            tf.summary.scalar("activation_mean", tf.reduce_mean(a2_u))
            tf.summary.scalar("activation_variance", tf.reduce_mean(a2_var))
            
        with tf.name_scope('conv_3') as scope:
        
            k_sz = 5 # the square kernel size
            in_ch = self.nb_filters * 2
            out_ch = self.nb_filters * 2
            
            #self.W_conv3 = self.initialize_kernel([k_sz, k_sz, in_ch, out_ch])
            with tf.variable_scope('conv_3_init', reuse=self.reuse):
                self.W_conv3 = self.get_kernel([k_sz, k_sz, in_ch, out_ch])
            
            # weight decay
            self.W_conv3_p = tf.nn.l2_loss(self.W_conv3)
            self.W_conv3_l1 = tf.reduce_mean(tf.abs(self.W_conv3))
    
            if self.use_batch_norm:
                with tf.variable_scope('conv_3_bn', reuse=self.reuse):
                    h_conv2 = tf.contrib.layers.batch_norm(h_conv2, is_training=phase)

            h_conv3 = tf.nn.relu(tf.nn.conv2d(
                h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID'))

            a3_u, a3_var = tf.nn.moments(
                tf.abs(h_conv3), axes=[0], keep_dims=False)

            tf.summary.histogram(name='W_conv3_summ', values=self.W_conv3)
            tf.summary.histogram(name='h_conv3_summ', values=h_conv3)
            tf.summary.scalar("activation_mean", tf.reduce_mean(a3_u))
            tf.summary.scalar("activation_variance", tf.reduce_mean(a3_var))
            
        with tf.name_scope('fc_out') as scope:

            nb_classes = 10
            in_sz = self.nb_filters * 8
            
            #W_fcout = self.initialize_weight([in_sz, nb_classes])
            with tf.variable_scope('fc_out_init', reuse=self.reuse):
                W_fcout = self.get_weight([in_sz, nb_classes])
            
            # weight decay
            self.W_fcout_p = tf.nn.l2_loss(W_fcout)
            self.W_fcout_l1 = tf.reduce_mean(tf.abs(W_fcout))

            h_conv3_flat = tf.reshape(h_conv3, [-1, in_sz])
            
            # tensor equivalent of numpy.dot()
            self.output = tf.matmul(h_conv3_flat, W_fcout) 

            y_u, y_var = tf.nn.moments(
                tf.abs(self.output), axes=[0], keep_dims=False)

            norm_out = tf.norm(W_fcout)
            
            tf.summary.histogram(name='output_summ', values=self.output)
            tf.summary.scalar("norm_out", norm_out)
            tf.summary.scalar("logits_mean", tf.reduce_mean(y_u))
            tf.summary.scalar("logits_var", tf.reduce_mean(y_var))

def data_cifar10():
    """
    Preprocess CIFAR-10 dataset
    :return:
    """

    # These values are specific to CIFAR-10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test

train_x, train_y, test_x, test_y = data_cifar10()
__, img_rows, img_cols, channels = train_x.shape
__, nb_classes = train_y.shape

# used to keep track of how many steps we've trained our model for
tf.reset_default_graph()
global_step = tf.Variable(0, trainable=False)

dtype = tf.float32
x_ = tf.placeholder(dtype, shape=(None, img_rows, img_cols, channels))
y_ = tf.placeholder(dtype, shape=(None, nb_classes))

# This is for batch normalization. True means training mode, False means testing mode.
phase = tf.placeholder_with_default(True, shape=(), name='phase')

def build_model_save_path(root_path, batch_size, nb_filters, learning_rate, epochs):
    
    model_path = os.path.join(root_path, 'k_' + str(nb_filters))
    model_path = os.path.join(model_path, 'bs_' + str(batch_size))
    model_path = os.path.join(model_path, 'lr_%1.e' % learning_rate)
    model_path = os.path.join(model_path, 'ep_' + str(epochs))
    '''
    optionally create this folder if it does not already exist,
    otherwise, increment the subfolder number
    '''
    model_path = create_dir_if_not_exists(model_path)

    return model_path


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        path += '/1'
        os.makedirs(path)
    else:
        digits = []
        sub_dirs = next(os.walk(path))[1]
        [digits.append(s) for s in sub_dirs if s.isnumeric()]
        if len(digits) > 0:
            sub = str(np.max(np.asarray(sub_dirs).astype('uint8')) + 1)        
        else:
            sub = '1'
        path = os.path.join(path, sub)
        os.makedirs(path)
    print('Logging to:%s' % path)
    return path


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

l2_reg = 1e-3
l1_reg = 1e-3
nb_epochs = 25
nb_filters = 64
batch_size = 128 # normally use 128
learning_rate = 1e-3
batch_norm = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = '/scratch/ssd/logs/cifar10'

# assume we're going to train from scratch, and not log checkpoints
save = False
train_from_scratch = True

if model_path is not None:
    if os.path.exists(model_path):
        # check for existing model in immediate subfolder
        if any(f.endswith('.meta') for f in os.listdir(model_path)):
            train_from_scratch = False
        else:
            save = True
            model_path = build_model_save_path(
                model_path, batch_size, nb_filters, learning_rate, nb_epochs)

cnn = ConvNet(x_, nb_filters, batch_norm, phase, reuse=tf.AUTO_REUSE) 

logits = cnn.output

total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=logits))

# if you just want the model predictions, use the following:
# preds = tf.nn.softmax(logits)

total_loss += l2_reg * (cnn.W_conv1_p + cnn.W_conv2_p +
                        cnn.W_conv3_p + cnn.W_fcout_p)

total_loss += l1_reg * (cnn.W_conv1_l1 + cnn.W_conv2_l1 +
                        cnn.W_conv3_l1 + cnn.W_fcout_l1)

if batch_norm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # ensures that we execute the update_ops before performing the train_op
        train_op = tf.contrib.layers.optimize_loss(
                    total_loss, global_step, learning_rate=learning_rate, optimizer='Adam',  # SGD
                    summaries=["gradients"])
else:
    train_op = tf.contrib.layers.optimize_loss(
                total_loss, global_step, learning_rate=learning_rate, optimizer='Adam',
                summaries=["gradients"])

'''
create tensors that will automatically compute the number of 
correct predictions and accuracy in a sample
'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)

# sess = tf.Session()
start_time = time.time()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
d = float(time.time() - start_time)
print("Startup time was %.4f" % d)

# setup summary writer
if save:
    summary_writer = tf.summary.FileWriter(model_path, sess.graph)
    tf.summary.scalar(
                "stats/train_loss", total_loss)
    tf.summary.scalar("stats/train_accuracy", accuracy)
    
    # create one op that will run all summaries
    merge_op = tf.summary.merge_all()
    checkpoint_path = os.path.join(model_path, 'model.ckpt')

def evaluate(sess, tensor, x, y, x_np, y_np, feed=None):
    feed_dict = {x: x_np, y: y_np, phase: False}
    if feed is not None:
        feed_dict.update(feed)
    return sess.run(tensor, feed_dict)

def evaluate_model(sess, accuracy, x, y, test_x, test_y, batch_size):
    """
    This helper function evaluates a model on one pass through
    the test set
    :param accuracy: the tensor that computes accuracy
    :param x: input placeholder
    :param y: output placeholder
    :param test_x: the test examples
    :param test_y: the test labels
    :param batch_size: batch size to use when evaluating
    :return: accuracy on the test set
    """
    nb_test_examples = test_x.shape[0]
    nb_test_batches = int(
        np.ceil(float(nb_test_examples) / batch_size))
    # print('nb_test_batches=%d' % nb_test_batches)
    assert nb_test_batches * batch_size >= nb_test_examples

    tot_accuracy = 0.0
    for e, test_batch in enumerate(range(nb_test_batches)):
        # Must not use the `batch_indices` function here, because it
        # repeats some examples.
        # It's acceptable to repeat during training, but not eval.
        start = test_batch * batch_size
        end = min(nb_test_examples, start + batch_size)
        cur_batch_size = end - start
        batch_xs = test_x[start:end]
        batch_ys = test_y[start:end]
        cur_acc = evaluate(sess, accuracy, x,
                           y_, batch_xs, batch_ys)
        tot_accuracy += (cur_batch_size * cur_acc)
    tot_accuracy /= nb_test_examples
    return tot_accuracy

if train_from_scratch:
    step = 0
    init_step = 0
    max_acc = 0
    
    # Compute number of training batches
    nb_training_examples = train_x.shape[0]
    nb_batches = int(
    np.ceil(float(nb_training_examples) / batch_size))
    print('nb_training_batches=%d' % nb_batches)
    assert nb_batches * batch_size >= nb_training_examples

    for epoch in range(nb_epochs):

        # Indices to shuffle training set
        index_shuf = np.arange(nb_training_examples)
        np.random.shuffle(index_shuf)

        for batch in tqdm(range(nb_batches)):
            
            start_time = time.time()
            step = init_step + (epoch * nb_batches + batch)

            # Compute batch start and end indices
            start, end = batch_indices(
                batch, nb_training_examples, batch_size)

            batch_xs = train_x[index_shuf[start:end]]
            batch_ys = train_y[index_shuf[start:end]]

            __, loss_val, summ = sess.run([train_op, total_loss, merge_op], feed_dict={
                x_: batch_xs, y_: batch_ys})
            duration = time.time() - start_time
            summary_writer.add_summary(summ, global_step=step)
        summary_writer.flush()

        # Init result var
        tot_accuracy = evaluate_model(
            sess, accuracy, x_, y_, test_x, test_y, batch_size)       

        print("epoch %d, loss=%.4f, test_acc=%.4f (%.1f ex/s)" %
              (epoch, loss_val, tot_accuracy, float(batch_size / duration)))

        if model_path:
            saver.save(sess, checkpoint_path, global_step=step)
        step += 1
    # close the TensorFlow client session
    tf.reset_default_graph()
    sess.close() 





