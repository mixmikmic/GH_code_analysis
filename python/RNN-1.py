import pickle
import tensorflow as tf
import numpy as np

from nov20.prepare_data import parse_seq

tf.reset_default_graph()

seq_len = 200
summaries_dir = 'nov20/summaries/shakespeare_{}'.format(seq_len)
prefix = 'nov20/shakespeare/seq{}'.format(seq_len)
seq_file = prefix + '.tfrecords'
vocab_file = prefix + '_vocab'

batch_size = 100
num_epochs = 100

with open(vocab_file, 'rb') as fin:
    ch_to_idx = pickle.load(fin)
    num_chars = len(ch_to_idx)    

dataset = tf.contrib.data.TFRecordDataset([seq_file])
dataset = dataset.map(lambda x: parse_seq(x, seq_len=seq_len))
dataset = dataset.map(lambda x: tf.one_hot(x, num_chars))
dataset = dataset.shuffle(1000).batch(batch_size).repeat(num_epochs)

iterator = dataset.make_one_shot_iterator()

hidden_state_size = 512

init_state = tf.placeholder(tf.float32, [None, hidden_state_size])
hidden_state = init_state

with tf.name_scope("input"):
    x_seq = iterator.get_next()
    
with tf.variable_scope("hidden"):
    W_xh = tf.get_variable('W_xh', [num_chars, hidden_state_size])
    B_xh = tf.get_variable('B_xh', [hidden_state_size])

    W_hh = tf.get_variable('W_hh', [hidden_state_size, hidden_state_size])
    B_hh = tf.get_variable('B_hh', [1, hidden_state_size])

with tf.variable_scope("output"):
    W_hy = tf.get_variable('W_hy', [hidden_state_size, num_chars])
    B_hy = tf.get_variable('B_hy', [num_chars])
    

# unroll computation graph
costs = []
for i in range(seq_len - 1):
    x = x_seq[:, i, :]
    y_ = x_seq[:, i+1, :]

    hidden_state = tf.nn.tanh(tf.matmul(x, W_xh) + tf.matmul(hidden_state, W_hh) + B_hh)
    y = tf.matmul(hidden_state, W_hy) + B_hy

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=y))
    costs.append(cross_entropy)

total_cost = tf.reduce_mean(costs)
train_op = tf.train.AdamOptimizer().minimize(total_cost)

tf.summary.scalar('total_cost', total_cost)
summary_op = tf.summary.merge_all()    

saver = tf.train.Saver()
with tf.Session() as sesh:
    writer = tf.summary.FileWriter(summaries_dir, sesh.graph)
    sesh.run(tf.global_variables_initializer())
    
    for step in range(5000+1):
        _ = sesh.run([train_op],
                     feed_dict={init_state: np.zeros((1, hidden_state_size))})
        if step % 500 == 0:
            xent, summary, _ = sesh.run([total_cost, summary_op, train_op],
                     feed_dict={init_state: np.zeros((1, hidden_state_size))})
            writer.add_summary(summary, step)
            print("Step {}\tcost {}".format(step, xent))
        if step % 1000 == 0:
            saver.save(sesh, summaries_dir + "/model.ckpt", global_step=step)
writer.close()

seq_len = 200
summaries_dir = 'nov20/summaries/shakespeare_{}'.format(seq_len)
prefix = 'nov20/shakespeare/seq{}'.format(seq_len)
seq_file = prefix + '.tfrecords'
vocab_file = prefix + '_vocab'

batch_size = 100
num_epochs = 100

with open(vocab_file, 'rb') as fin:
    ch_to_idx = pickle.load(fin)
    num_chars = len(ch_to_idx)
idx_to_ch = {v: k for k, v in ch_to_idx.items()}
    
hidden_state_size = 512

tf.reset_default_graph()

model_path = summaries_dir + "/model.ckpt-5000"

input_char = tf.placeholder(tf.float32, [None, num_chars])
input_state = tf.placeholder(tf.float32, [None, hidden_state_size])
hidden_state = input_state
    
with tf.variable_scope("hidden"):
    W_xh = tf.get_variable('W_xh', [num_chars, hidden_state_size])
    B_xh = tf.get_variable('B_xh', [hidden_state_size])

    W_hh = tf.get_variable('W_hh', [hidden_state_size, hidden_state_size])
    B_hh = tf.get_variable('B_hh', [1, hidden_state_size])

with tf.variable_scope("output"):
    W_hy = tf.get_variable('W_hy', [hidden_state_size, num_chars])
    B_hy = tf.get_variable('B_hy', [num_chars])
    

hidden_state = tf.nn.tanh(tf.matmul(input_char, W_xh) + tf.matmul(hidden_state, W_hh) + B_hh)
y = tf.matmul(hidden_state, W_hy) + B_hy

probs_op = tf.nn.softmax(logits=y)

output = ''
saver = tf.train.Saver()

with tf.Session() as sesh:
    saver.restore(sesh, model_path)
    cur_char = np.zeros((1, num_chars))
    cur_char[0, 0] = 1.
    
    cur_state = np.zeros((1, hidden_state_size))
    
    for _ in range(500):
        probs, next_state = sesh.run([probs_op, hidden_state], feed_dict={
            input_char: cur_char,
            input_state: cur_state
        })
        probs = np.squeeze(probs)
        next_char_pos = np.random.choice(num_chars, p=probs)
        output +=  idx_to_ch[next_char_pos]
        
        cur_char = np.zeros((1, num_chars))
        cur_char[0, next_char_pos] = 1.
        cur_state = next_state

print(output)    



