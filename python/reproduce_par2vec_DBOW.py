#!pip install nltk
import nltk
#nltk.download()

from reproduce_par2vec_commons import *

orig_labels = get_labels()

dictionary, vocab_size, data, doclens = build_dictionary()

twcp = get_text_window_center_positions(data)
print len(twcp)
np.random.shuffle(twcp)
twcp_train_gen = repeater_shuffler(twcp)
del twcp # save some memory

def create_training_graph():
    # Input data
    dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
    # Weights
    embeddings = tf.Variable(
        tf.random_uniform([len(doclens), EMBEDDING_SIZE],
                          -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal(
            [vocab_size, EMBEDDING_SIZE],
            stddev=1.0 / np.sqrt(EMBEDDING_SIZE)))
    softmax_biases = tf.Variable(tf.zeros([vocab_size]))
    # Model
    # Look up embeddings for inputs
    embed = tf.nn.embedding_lookup(embeddings, dataset)
    # Compute the softmax loss, using a sample of the negative
    # labels each time
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            softmax_weights, softmax_biases, labels,
            embed, NUM_SAMPLED, vocab_size))
    # Optimizer
    optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(
        loss)
    # Normalized embeddings (to use cosine similarity later on)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1,
                                 keep_dims=True))
    normalized_embeddings = embeddings / norm
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    return dataset, labels, softmax_weights, softmax_biases, loss, optimizer, normalized_embeddings, session

def generate_batch_single_twcp(twcp, i, batch, labels):
    tw_start = twcp - (TEXT_WINDOW_SIZE - 1) // 2
    tw_end = twcp + TEXT_WINDOW_SIZE // 2 + 1
    docids, wordids = zip(*data[tw_start:tw_end])
    batch_slice = slice(i * TEXT_WINDOW_SIZE,
                        (i + 1) * TEXT_WINDOW_SIZE)
    batch[batch_slice] = docids
    labels[batch_slice, 0] = wordids


def generate_batch(twcp_gen):
    batch = np.ndarray(shape=(BATCH_SIZE,), dtype=np.int32)
    labels = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)
    for i in range(BATCH_SIZE // TEXT_WINDOW_SIZE):
        generate_batch_single_twcp(next(twcp_gen), i, batch, labels)
    return batch, labels

def train(optimizer, loss, dataset, labels):
    avg_training_loss = 0
    for step in range(NUM_STEPS):
        batch_data, batch_labels = generate_batch(twcp_train_gen)
        _, l = session.run(
                [optimizer, loss],
                feed_dict={dataset: batch_data, labels: batch_labels})
        avg_training_loss += l
        if step > 0 and step % REPORT_EVERY_X_STEPS == 0:
            avg_training_loss =                     avg_training_loss / REPORT_EVERY_X_STEPS
            # The average loss is an estimate of the loss over the
            # last REPORT_EVERY_X_STEPS batches
            print('Average loss at step {:d}: {:.1f}'.format(
                    step, avg_training_loss))

dataset, labels, softmax_weights, softmax_biases, loss, optimizer, normalized_embeddings, session = create_training_graph()
train(optimizer, loss, dataset, labels)
current_embeddings = session.run(normalized_embeddings)
current_softmax_weights = session.run(softmax_weights)
current_softmax_biases = session.run(softmax_biases)

def test(doc):
    test_data, test_twcp = build_test_twcp(doc, dictionary)

    session, test_dataset, test_labels, test_loss, test_normalized_embedding, test_optimizer, test_softmax_biases, test_softmax_weights = create_test_graph(
        test_twcp)

    for step in range(NUM_STEPS):
        test_input = np.ndarray(shape=(TEXT_WINDOW_SIZE*len(test_twcp),), dtype=np.int32)
        test_labels_values = np.ndarray(shape=(TEXT_WINDOW_SIZE*len(test_twcp), 1), dtype=np.int32)
        i = 0
        for twcp in test_twcp:
            tw_start = twcp - (TEXT_WINDOW_SIZE - 1) // 2
            tw_end = twcp + TEXT_WINDOW_SIZE // 2 + 1
            docids, wordids = zip(*test_data[tw_start:tw_end])
            batch_slice = slice(i * TEXT_WINDOW_SIZE,
                                (i + 1) * TEXT_WINDOW_SIZE)
            test_input[batch_slice] = docids
            test_labels_values[batch_slice, 0] = wordids
            i += 1
        _, l = session.run(
                [test_optimizer, test_loss],
                feed_dict={test_dataset: test_input, test_labels: test_labels_values,
                           test_softmax_weights: current_softmax_weights,
                           test_softmax_biases: current_softmax_biases})
    current_test_embedding = session.run(test_normalized_embedding)
    return current_test_embedding


def create_test_graph(test_twcps):
    # Input data
    test_dataset = tf.placeholder(tf.int32, shape=[TEXT_WINDOW_SIZE * len(test_twcps)])
    test_labels = tf.placeholder(tf.int32, shape=[TEXT_WINDOW_SIZE * len(test_twcps), 1])
    test_softmax_weights = tf.placeholder(tf.float32, shape=[vocab_size, EMBEDDING_SIZE])
    test_softmax_biases = tf.placeholder(tf.float32, shape=[vocab_size])
    # Weights
    test_embedding = tf.Variable(
        tf.random_uniform([1, EMBEDDING_SIZE],
                          -1.0, 1.0))
    # Model
    # Look up embeddings for inputs
    test_embed = tf.nn.embedding_lookup(test_embedding, test_dataset)
    # Compute the softmax loss, using a sample of the negative
    # labels each time
    test_loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            test_softmax_weights, test_softmax_biases, test_labels,
            test_embed, NUM_SAMPLED, vocab_size))
    # Optimizer
    test_optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(
        test_loss)
    # Normalized embeddings (to use cosine similarity later on)
    test_norm = tf.sqrt(tf.reduce_sum(tf.square(test_embedding), 1,
                                      keep_dims=True))
    test_normalized_embedding = test_embedding / test_norm
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    return session, test_dataset, test_labels, test_loss, test_normalized_embedding, test_optimizer, test_softmax_biases, test_softmax_weights

test_embedding_1 = test('something cringe-inducing about seeing an American football stadium nuked as pop entertainment')
test_embedding_2 = test('something cringe-inducing about seeing an American football stadium nuked as pop entertainment')
distance = spatial.distance.cosine(test_embedding_1, test_embedding_2)
print distance

test_logistic_regression(current_embeddings, orig_labels)



