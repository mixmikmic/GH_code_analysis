#!pip install nltk
#import nltk
# download punkt package
#nltk.download()

from reproduce_par2vec_commons import *

orig_labels = get_labels()

dictionary, vocab_size, data, doclens = build_dictionary()

twcp = get_text_window_center_positions(data)
print len(twcp)
np.random.shuffle(twcp)
twcp_train_gen = repeater_shuffler(twcp)
del twcp  # save some memory

def create_training_graph():
    # Input data
    dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE, TEXT_WINDOW_SIZE])
    labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
    # Variables.
    # embeddings for words, W in paper
    word_embeddings = tf.Variable(
        tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0))
    # embedding for documents (can be sentences or paragraph), D in paper
    doc_embeddings = tf.Variable(
        tf.random_uniform([len(doclens), EMBEDDING_SIZE], -1.0, 1.0))
    combined_embed_vector_length = EMBEDDING_SIZE * TEXT_WINDOW_SIZE
    # softmax weights, W and D vectors should be concatenated before applying softmax
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocab_size, combined_embed_vector_length],
                            stddev=1.0 / np.math.sqrt(combined_embed_vector_length)))
    # softmax biases
    softmax_biases = tf.Variable(tf.zeros([vocab_size]))
    # Model.
    # Look up embeddings for inputs.
    # shape: (batch_size, embeddings_size)
    embed = []  # collect embedding matrices with shape=(batch_size, embedding_size)
    for j in range(TEXT_WINDOW_SIZE - 1):
        embed_w = tf.nn.embedding_lookup(word_embeddings, dataset[:, j])
        embed.append(embed_w)
    embed_d = tf.nn.embedding_lookup(doc_embeddings, dataset[:, TEXT_WINDOW_SIZE - 1])
    embed.append(embed_d)
    # concat word and doc vectors
    embed = tf.concat(embed, 1)
    # Compute the softmax loss, using a sample of the negative
    # labels each time
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            softmax_weights, softmax_biases, labels,
            embed, NUM_SAMPLED, vocab_size))
    # Optimizer
    optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(
        loss)
    # We use the cosine distance:
    norm_w = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
    normalized_word_embeddings = word_embeddings / norm_w
    norm_d = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
    normalized_doc_embeddings = doc_embeddings / norm_d
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    return optimizer, loss, dataset, labels,           normalized_word_embeddings,            normalized_doc_embeddings,            session, softmax_weights, softmax_biases

def generate_batch_single_twcp(twcp, i, batch, labels):
    tw_start = twcp - (TEXT_WINDOW_SIZE - 1) // 2
    tw_end = twcp + TEXT_WINDOW_SIZE // 2 + 1
    docids, wordids = zip(*data[tw_start:tw_end])

    wordids_list = list(wordids)
    twcp_index = (TEXT_WINDOW_SIZE - 1) // 2
    twcp_docid = data[twcp][0]
    twcp_wordid = data[twcp][1]
    del wordids_list[twcp_index]
    wordids_list.append(twcp_docid)

    batch[i] = wordids_list
    labels[i] = twcp_wordid


def generate_batch(twcp_gen):
    batch = np.ndarray(shape=(BATCH_SIZE, TEXT_WINDOW_SIZE), dtype=np.int32)
    labels = np.ndarray(shape=(BATCH_SIZE, 1), dtype=np.int32)
    for i in range(BATCH_SIZE):
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
            avg_training_loss =                 avg_training_loss / REPORT_EVERY_X_STEPS
            # The average loss is an estimate of the loss over the
            # last REPORT_EVERY_X_STEPS batches
            print('Average loss at step {:d}: {:.1f}'.format(
                step, avg_training_loss))

optimizer, loss, dataset, labels, word_embeddings, doc_embeddings, session, softmax_weights, softmax_biases = create_training_graph()
train(optimizer, loss, dataset, labels)
current_embeddings = session.run(doc_embeddings)
current_word_embeddings = session.run(word_embeddings)
current_softmax_weights = session.run(softmax_weights)
current_softmax_biases = session.run(softmax_biases)

def test(doc, train_word_embeddings, train_softmax_weights, train_softmax_biases):
    test_data, test_twcp = build_test_twcp(doc, dictionary)
    # Input data
    combined_embed_vector_length = EMBEDDING_SIZE * TEXT_WINDOW_SIZE
    test_dataset = tf.placeholder(tf.int32, shape=[len(test_twcp), TEXT_WINDOW_SIZE])
    test_labels = tf.placeholder(tf.int32, shape=[len(test_twcp), 1])
    test_softmax_weights = tf.placeholder(tf.float32, shape=[vocab_size, combined_embed_vector_length])
    test_softmax_biases = tf.placeholder(tf.float32, shape=[vocab_size])
    test_word_embeddings = tf.placeholder(tf.float32, shape=[vocab_size, EMBEDDING_SIZE])
    # Variables.
    # embedding for documents (can be sentences or paragraph), D in paper
    test_doc_embeddings = tf.Variable(
        tf.random_uniform([1, EMBEDDING_SIZE], -1.0, 1.0))

    # Look up embeddings for inputs.
    # shape: (batch_size, embeddings_size)
    test_embed = []  # collect embedding matrices with shape=(batch_size, embedding_size)
    for j in range(TEXT_WINDOW_SIZE - 1):
        test_embed_w = tf.gather(test_word_embeddings, test_dataset[:,j])
        test_embed.append(test_embed_w)
    test_embed_d = tf.nn.embedding_lookup(test_doc_embeddings, test_dataset[:, TEXT_WINDOW_SIZE - 1])
    test_embed.append(test_embed_d)
    # concat word and doc vectors
    test_embed = tf.concat(test_embed, 1)
    # Compute the softmax loss, using a sample of the negative
    # labels each time
    test_loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            test_softmax_weights, test_softmax_biases, test_labels,
            test_embed, NUM_SAMPLED, vocab_size))
    # Optimizer
    test_optimizer = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(
        test_loss)
    # We use the cosine distance:
    test_norm_d = tf.sqrt(tf.reduce_sum(tf.square(test_doc_embeddings), 1, keep_dims=True))
    test_normalized_doc_embeddings = test_doc_embeddings / test_norm_d
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for step in range(NUM_STEPS):
        test_input = np.ndarray(shape=(len(test_twcp), TEXT_WINDOW_SIZE), dtype=np.int32)
        labels_values = np.ndarray(shape=(len(test_twcp), 1), dtype=np.int32)
        i = 0
        for twcp in test_twcp:
            tw_start = twcp - (TEXT_WINDOW_SIZE - 1) // 2
            tw_end = twcp + TEXT_WINDOW_SIZE // 2 + 1
            docids, wordids = zip(*test_data[tw_start:tw_end])

            wordids_list = list(wordids)
            twcp_index = (TEXT_WINDOW_SIZE - 1) // 2
            twcp_docid = test_data[twcp][0]
            twcp_wordid = test_data[twcp][1]
            del wordids_list[twcp_index]
            wordids_list.append(twcp_docid)

            test_input[i] = wordids_list
            labels_values[i] = twcp_wordid
            i += 1
        _, l = session.run(
            [test_optimizer, test_loss],
            feed_dict={test_dataset: test_input, test_labels: labels_values,
                       test_word_embeddings: train_word_embeddings,
                       test_softmax_weights: train_softmax_weights,
                       test_softmax_biases: train_softmax_biases
                       })
    current_test_embedding = session.run(test_normalized_doc_embeddings)
    return current_test_embedding

test_embedding_1 = test('something cringe-inducing about seeing an American football stadium nuked as pop entertainment',
                        current_word_embeddings, current_softmax_weights, current_softmax_biases)
test_embedding_2 = test('something cringe-inducing about seeing an American football stadium nuked as pop entertainment',
                        current_word_embeddings, current_softmax_weights, current_softmax_biases)
distance = spatial.distance.cosine(test_embedding_1, test_embedding_2)
print distance

test_logistic_regression(current_embeddings, orig_labels)



