import pickle
import tensorflow as tf
import numpy as np

from nov27.prepare_data import parse_seq

tf.reset_default_graph()

hidden_state_size = 256
summaries_dir = 'nov27/summaries/{}'.format(hidden_state_size)

max_seq_len = 100
prefix = 'nov27/bible/kj{}'.format(max_seq_len)
seq_file = prefix + '.tfrecords'
vocab_file = prefix + '_vocab'

batch_size = 100
num_epochs = 100

with open(vocab_file, 'rb') as fin:
    ch_to_idx = pickle.load(fin)
    num_chars = len(ch_to_idx)    

dataset = tf.contrib.data.TFRecordDataset([seq_file])
dataset = dataset.map(parse_seq)
dataset = dataset.shuffle(1000).repeat(num_epochs).padded_batch(batch_size, [None])

iterator = dataset.make_one_shot_iterator()

cell_state = tf.placeholder(tf.float32, [batch_size, hidden_state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

with tf.name_scope("input"):
    inputs = iterator.get_next()

xs = inputs[:, :-1]
valid_xs_mask = tf.not_equal(xs, 0)

xs_seq_len = tf.reduce_sum(tf.to_int32(valid_xs_mask), axis=1)
one_hot_xs = tf.one_hot(xs, depth=num_chars)
ys = inputs[:, 1:]
one_hot_ys = tf.one_hot(ys, depth=num_chars)
    
rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_state_size, state_is_tuple=True)
outputs, state = tf.nn.dynamic_rnn(rnn_cell, one_hot_xs, sequence_length=xs_seq_len, initial_state=init_state)

with tf.variable_scope("output"):
    W_hy = tf.get_variable('W_hy', [hidden_state_size, num_chars])
    B_hy = tf.get_variable('B_hy', [num_chars])
    
logits = tf.tensordot(outputs, W_hy, axes=1) + B_hy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_ys, logits=logits)
masked_cross_entropy = tf.multiply(cross_entropy, tf.to_float(valid_xs_mask))
summed_entropy = tf.reduce_sum(masked_cross_entropy, 1)
sequence_entropy = tf.divide(summed_entropy, tf.to_float(xs_seq_len))
total_entropy = tf.reduce_mean(sequence_entropy)

train_op = tf.train.AdamOptimizer().minimize(total_entropy)

tf.summary.scalar('entropy', total_entropy)
summary_op = tf.summary.merge_all()  

saver = tf.train.Saver()
with tf.Session() as sesh:
    writer = tf.summary.FileWriter(summaries_dir, sesh.graph)
    sesh.run(tf.global_variables_initializer())
    
    for step in range(5000+1):
        _ = sesh.run([train_op], feed_dict={
                         hidden_state: np.zeros([batch_size, hidden_state_size]),
                         cell_state: np.zeros([batch_size, hidden_state_size])
                     })
        if step % 100 == 0:
            cost, summary, _ = sesh.run([total_entropy, summary_op, train_op], feed_dict={
                hidden_state: np.zeros((batch_size, hidden_state_size)),
                cell_state: np.zeros((batch_size, hidden_state_size))})
            
            writer.add_summary(summary, step)
            print("Step {}\tcost {}".format(step, cost))
        if step % 1000 == 0:
            saver.save(sesh, summaries_dir + "/model.ckpt", global_step=step)
writer.close()

tf.reset_default_graph()

summaries_dir = 'nov27/summaries/{}/'.format(hidden_state_size)
max_seq_len = 100
prefix = 'nov27/bible/kj{}'.format(max_seq_len)
seq_file = prefix + '.tfrecords'
vocab_file = prefix + '_vocab'

batch_size = 100
num_epochs = 100

with open(vocab_file, 'rb') as fin:
    ch_to_idx = pickle.load(fin)
    num_chars = len(ch_to_idx) 
    idx_to_ch = {v: k for k, v in ch_to_idx.items()}

start_char = '<S>'
stop_char = '</S>'
    
dataset = tf.contrib.data.TFRecordDataset([seq_file])
dataset = dataset.map(parse_seq)
dataset = dataset.shuffle(1000).repeat(num_epochs).padded_batch(batch_size, [None])

iterator = dataset.make_one_shot_iterator()

cell_state = tf.placeholder(tf.float32, [None, hidden_state_size])
hidden_state = tf.placeholder(tf.float32, [None, hidden_state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

input_char = tf.placeholder(tf.float32, [None, None, num_chars])
    
rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_state_size, state_is_tuple=True)
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_char, sequence_length=[1], initial_state=init_state)

with tf.variable_scope("output"):
    W_hy = tf.get_variable('W_hy', [hidden_state_size, num_chars])
    B_hy = tf.get_variable('B_hy', [num_chars])
    
logits = tf.tensordot(outputs, W_hy, axes=1) + B_hy
probs_op = tf.nn.softmax(logits=logits)

model_path = summaries_dir + "model.ckpt-5000"

output = ''
saver = tf.train.Saver()

with tf.Session() as sesh:
    saver.restore(sesh, model_path)
    
    # init
    cur_char = start_char
    cur_char_vec = np.zeros((1, 1, num_chars))
    cur_char_pos = ch_to_idx[start_char]
    cur_char_vec[0, 0, cur_char_pos] = 1.
    
    cur_hidden_state = np.zeros((1, hidden_state_size))
    cur_output_state= np.zeros((1, hidden_state_size))
    
    while True:
        probs, cur_state = sesh.run([probs_op, state], feed_dict={
            input_char: cur_char_vec,
            cell_state: cur_output_state,
            hidden_state: cur_hidden_state,
        })
        probs = np.squeeze(probs)
        cur_char_pos = np.random.choice(num_chars, p=probs)
        cur_char = idx_to_ch[cur_char_pos]
        
        if cur_char == stop_char:
            break
            
        output +=  cur_char
        
        cur_char_vec = np.zeros((1, 1, num_chars))
        cur_char_vec[0, 0, cur_char_pos] = 1.
        cur_output_state, cur_hidden_state = cur_state
            
print(output[:-1])

