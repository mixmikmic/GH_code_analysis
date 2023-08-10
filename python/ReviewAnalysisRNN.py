import tensorflow as tf
import numpy as np
import time
from collections import Counter

reviews = ''
labels = ''

with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
    
with open('data/labels.txt', 'r') as f:
    labels = f.read()

print ('Sample Positive Review-->')
print (reviews.split('\n')[0])

print ('Sample Negative Review-->')
print (reviews.split('\n')[1])

len(labels.split('\n'))

total_words = len(reviews.split())
total_characters = len(reviews)
unique_words = len(set(reviews.split()))
unique_characters = len(set(reviews))

print ('FOR REVIEWS')
print ("Total words :", total_words)
print ("Total characters :", total_characters)
print ("Unique words :", unique_words)
print ("Unique characters:", unique_characters)

from string import punctuation

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')
labels = labels.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

len(labels)

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for i in range(len(reviews)):
    if(labels[i] == 'positive'):
        for word in reviews[i].split():
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split():
            negative_counts[word] += 1
            total_counts[word] += 1

positive_counts.most_common()[:20]

negative_counts.most_common()[:20]

pos_neg_ratios = Counter()
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

for word,ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[word] = np.log(ratio)

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

pos_neg_ratios.most_common()[:30]

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])

vocab_to_int

print (reviews[0])
x = []
for i in reviews[0].split():
    x.append(vocab_to_int[i])
print ('\n')    
print (x)    

labels = np.array([1 if each == 'positive' else 0 for each in labels])

non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
len(non_zero_idx)

review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])

len(reviews_ints)

seq_length = 300
features = np.zeros((len(reviews_ints), seq_length), dtype=int)
for i, row in enumerate(np.array(reviews_ints)):
    features[i, -len(row):] = np.array(row)[:seq_length]

features[:2,:]

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("Train set:{}".format(train_x.shape), 
      "\nValidation set:{}".format(val_x.shape),
      "\nTest set: {}".format(test_x.shape))

lstm_size = 512
lstm_layers = 2
batch_size = 64
learning_rate = 0.001
n_words = len(vocab_to_int) + 1
embed_size = 300

def placeholders():
    inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None), name='targets')
    keep_prob = tf.placeholder(tf.float32, name= 'keep_prob')
    
    return inputs, targets, keep_prob

def create_embedding(n_words, embed_size, inputs):
    embedding_matrix = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding_matrix, inputs)
    
    return embedding_matrix, embed

def lstm_cell(lstm_size, lstm_layers, batch_size, keep_prob):
    
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(lstm_layers)])
    state = cell.zero_state(batch_size, tf.float32)
    
    return cell, state

def generate_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

tf.reset_default_graph()

## Getting input tensors
inputs, targets, keep_prob = placeholders()

embedding_matrix, embed = create_embedding(n_words, embed_size, inputs)

## Creating LSTM Cell
cell, initial_state = lstm_cell(lstm_size, lstm_layers, batch_size, keep_prob)

## Collect outputs(RNN Forward Pass)
outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state = initial_state)

## Predictions
predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn = tf.sigmoid)

## Cost - Mean Squared Error
cost = tf.losses.mean_squared_error(targets, predictions)

## Gradient Descent Step - Backpropagation
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

## Calculating accuracy and correct predictions
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), targets)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

epochs = 10

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(generate_batches(train_x, train_y, batch_size), 1):
            feed = {inputs: x,
                    targets: y,
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in generate_batches(val_x, val_y, batch_size):
                    feed = {inputs: x,
                            targets: y,
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/sentiment.ckpt")

test_acc = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))



