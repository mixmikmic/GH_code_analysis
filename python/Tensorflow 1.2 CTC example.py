import tensorflow as tf
import numpy as np

# Convert dense tensor to sparse tensor, required for ctc
def to_sparse(tensor, lengths):
    mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
    indices = tf.to_int64(tf.where(tf.equal(mask, True)))
    values = tf.to_int32(tf.boolean_mask(tensor, mask))
    shape = tf.to_int64(tf.shape(tensor))
    return tf.SparseTensor(indices, values, shape)

vocab_size = 4
lstm_size = 10
embed_size = 10
samples = 100

# The max length of the label should be shorter than the min length of input
min_length = 4
max_length = 5
min_label_len = 2
max_label_len = 2

# Random inputs
inputs = tf.constant(np.random.randint(1, vocab_size, size=[samples, max_length]))
lengths = tf.constant(
    np.random.randint(min_length, max_length+1, size=samples),
    dtype=tf.int32)

# Random labels
labels = tf.constant(np.random.randint(1, vocab_size, size=[samples, max_label_len]))
label_lengths = tf.constant(
    np.random.randint(min_label_len, max_label_len+1, size=samples),
    dtype=tf.int32)

# Convert labels to sparse tensor
sparse_labels = to_sparse(labels, label_lengths)

# Transpose inputs to time-major
inputs = tf.transpose(inputs)

# Embed inputs
embed = tf.contrib.layers.embed_sequence(inputs, max_length, embed_size)

outputs, _ = tf.nn.dynamic_rnn(
    tf.nn.rnn_cell.LSTMCell(lstm_size),
    embed,
    lengths,
    time_major=True,
    dtype=tf.float32)

# Output layer converts lstm_size to vocab_size (plus one for blank label)
logits = tf.layers.dense(outputs, vocab_size + 1)

# Create train op from ctc loss
loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_labels, logits, lengths))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# Create test op from beam search decoder
decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, beam_width=2)
error_rate = tf.reduce_mean(tf.edit_distance(sparse_labels, tf.cast(decoded[0], tf.int32)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train_op)

    print(sess.run(error_rate))

