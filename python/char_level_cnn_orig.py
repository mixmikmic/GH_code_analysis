import tensorflow as tf
import numpy as np
import os
import time
import datetime
import json

try:
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        print("tqdm is notebook version")
        from tqdm import tqdm_notebook as tqdm
    else:
        raise RuntimeError
except (NameError, RuntimeError):
    from tqdm import tqdm
    
get_ipython().run_line_magic('load_ext', 'jupyternotify')

def extract_end(char_seq):
    char_seq_length = 600 #1014
    if len(char_seq) > char_seq_length:
        char_seq = char_seq[-char_seq_length:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 600
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            batch = list(zip(x_batch, y_batch))
            yield batch

dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 128
num_epochs = 50
evaluate_every = 5000
checkpoint_every = 1000

# Misc Parameters
allow_soft_placement = True #"Allow device soft device placement")
log_device_placement = False  #, "Log placement of ops on devices")

positive_data_file = "data/amazon/book_pos.txt"
negative_data_file = "data/amazon/book_neg.txt"

dev_sample_percentage = 0.0005

def _load_data(file_path, examples, raw):
    with open(file_path) as f:
        for line in tqdm(f):
            text = line
            raw.append(text)
            text_end_extracted = extract_end(list(text.lower()))
            padded = pad_sentence(text_end_extracted)
            text_int8_repr = string_to_int8_conversion(padded, alphabet)           
            examples.append(text_int8_repr)
    return examples, raw

def load_data(positive_file_path, negative_data_file, alphabet):
    pos_examples = [] 
    pos_raw = []
    neg_examples = []
    neg_raw = []
    
    _load_data(positive_data_file, pos_examples, pos_raw)
    _load_data(negative_data_file, neg_examples, neg_raw)
    
    n_posi = len(pos_examples)
    print("# of positive", n_posi)
    n_neg = len(neg_examples)
    print("# of negative", n_neg)
    
    positive_labels = [[0, 1] for _ in range(n_posi)]
    negative_labels = [[1, 0] for _ in range(n_neg)]
    
    labels = positive_labels + negative_labels
    examples = pos_examples + neg_examples
    
    return np.array(examples, dtype=np.int8), np.array(labels, dtype=np.int8), pos_raw+neg_raw

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

get_ipython().run_cell_magic('notify', '', 'x, y, raw = load_data(positive_data_file, negative_data_file, alphabet)')

x[1]

raw[1]

y[1]

x.shape

y.shape



length_list = np.array([len(r)for r in raw])
length_list.mean()

import pandas as pd
df = pd.DataFrame(length_list, columns=["length"])
df.head()

df.describe()

get_ipython().run_line_magic('matplotlib', 'inline')
df.hist(bins=50,figsize=(10,4))





get_ipython().run_cell_magic('notify', '', 'np.random.seed(10)\nshuffle_indices = np.random.permutation(np.arange(len(y)))\nx_shuffled = x[shuffle_indices]\ny_shuffled = y[shuffle_indices]\n\ndev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))\nx_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]\ny_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]\n\n#del x, y, x_shuffled, y_shuffled\n\nprint("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))')

class CharCNN(object):
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification paper.
    """
    def __init__(self, num_classes=2, filter_sizes=(7, 7, 3, 3, 3, 3), num_filters_per_size=256,
                 l2_reg_lambda=0.0, sequence_max_length=1014, num_quantized_chars=70):

        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    
        l2_loss = tf.constant(0.0)

        # ================ Layer 1 ================
        with tf.name_scope("conv-maxpool-1"):
            filter_shape = [num_quantized_chars, filter_sizes[0], 1, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID', name="pool1")

        # ================ Layer 2 ================
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [1, filter_sizes[1], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID', name="pool2")

        # ================ Layer 3 ================
        with tf.name_scope("conv-3"):
            filter_shape = [1, filter_sizes[2], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # ================ Layer 4 ================
        with tf.name_scope("conv-4"):
            filter_shape = [1, filter_sizes[3], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv4")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # ================ Layer 5 ================
        with tf.name_scope("conv-5"):
            filter_shape = [1, filter_sizes[4], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv5")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        # ================ Layer 6 ================
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [1, filter_sizes[5], num_filters_per_size, num_filters_per_size]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="VALID", name="conv6")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='VALID', name="pool6")

        # ================ Layer 7 ================
        num_features_total = num_filters_per_size * 18 #34
        print("######")
        print(num_features_total)
        print(pooled.shape)
        print("#############")
        h_pool_flat = tf.reshape(pooled, [-1, num_features_total])

        # Add dropout
        with tf.name_scope("dropout-1"):
            drop1 = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Fully connected layer 1
        with tf.name_scope("fc-1"):
            size = 64#1024
            W = tf.Variable(tf.truncated_normal([num_features_total, size], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(drop1, W, b), name="fc-1-out")

        # ================ Layer 8 ================
        # Add dropout
        with tf.name_scope("dropout-2"):
            drop2 = tf.nn.dropout(fc_1_output, self.dropout_keep_prob)

        # Fully connected layer 2
        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.truncated_normal([size, size], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[size]), name="b")
            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(drop2, W, b), name="fc-2-out")

        # ================ Layer 9 ================
        # Fully connected layer 3
        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.truncated_normal([size, num_classes], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            predictions = tf.argmax(scores, 1, name="predictions")
            
            

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss



        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

with tf.Graph().as_default() as g:
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        cnn = CharCNN(sequence_max_length=600)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)


        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            dev_size = len(x_batch)
            max_batch_size = 500
            num_batches = dev_size/max_batch_size
            acc = []
            losses = []
            print("Number of batches in dev set is " + str(num_batches))
            for i in range(num_batches):
                x_batch_dev, y_batch_dev = get_batched_one_hot(x_batch, y_batch, i * max_batch_size, (i + 1) * max_batch_size)
                feed_dict = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                acc.append(accuracy)
                losses.append(loss)
                time_str = datetime.datetime.now().isoformat()
                print("batch " + str(i + 1) + " in dev >>" +
                      " {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            print("\nMean accuracy=" + str(sum(acc)/len(acc)))
            print("Mean loss=" + str(sum(losses)/len(losses)))


        for batch in batch_iter(x_train, y_train, batch_size, num_epochs):
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))





