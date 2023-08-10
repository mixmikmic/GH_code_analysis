import os
import pickle


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))

data_dir = './data/blockchain.txt'
text = load_data(data_dir)

import numpy as np

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    words = set(text)
    vocab_to_int = {word: idx for idx, word in enumerate(words)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    res = {'.': "||Period||",
                       ',': "||Comma||", 
                      '"': "||Quotation_mark||", 
                      ';': "||Semicolon||", 
                      '!': "||Exclamation_mark||", 
                      '?': "||Question_mark||", 
                      '(': "||Left_Parentheses||", 
                      ')': "||Right_Parentheses||", 
                      '--': "||Dash||", 
                      '\n': "||Return||", 
                      '\t': "|||Tab|", 
                      '#': "||Hash_tag||", 
                      '$': "||Dollar||", 
                      '%': "||Percent||", 
                      '&': "||And||", 
                      '\'': "||Single_Quotation_mark||", 
                      '*': "||Asterisk||", 
                      '+': "||Plus||", 
                      '-': "||Minus||", 
                      '/': "||Slash||", 
                      '<': "||Less||", 
                      '=': "||Equal||", 
                      '>': "||Larger||", 
                      '@': "||At_Symbol||", 
                       '[': "||Left_bracket||", 
                       ']': "||Right_bracket||", 
                       '©': "||Copy_right_symbol||", 
                       'Ð': "||D_Symbol||", 
                      }
        
    return res

preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

import numpy as np

int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs_ = tf.placeholder(dtype = tf.int32, name='input', shape = [None, None])
    targets_ = tf.placeholder(dtype = tf.int32, name='target', shape = [None, None])
    learning_rate = tf.placeholder(name='lr', dtype = tf.float32)
    return inputs_, targets_, learning_rate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.8)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[dropout])
    initial_state = tf.identity(cell.zero_state(batch_size=batch_size, dtype=tf.float32), 
                                name = 'initial_state')
    return cell, initial_state

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embeddings = tf.Variable(
    tf.random_uniform([vocab_size, embed_dim], -1.0, 1.0))
    
    return tf.nn.embedding_lookup(embeddings, ids=input_data)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    rnn_out, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(inputs=rnn_out, num_outputs=vocab_size, activation_fn=None)
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text)//(batch_size * seq_length)
    res = np.zeros((n_batches, 2, batch_size, seq_length))
    
    total_len = n_batches *batch_size*seq_length
    inputs = np.array(int_text[:total_len])
    targets = int_text[1:total_len]
    targets.append(int_text[0])
    targets = np.array(targets)
    
    #shape: n_batches, 2, batch_size, seq_length
    
    skip = n_batches * seq_length
    for batch in range(n_batches):
        for n in range(batch_size):
            res[batch, 0, n] = inputs[range(batch * seq_length + n * skip, batch * seq_length + n * skip + seq_length)]
            res[batch, 1, n] = targets[range(batch * seq_length + n * skip, batch * seq_length + n * skip + seq_length)]
    
    return res

# Number of Epochs
num_epochs = 200
# Batch Size
batch_size = 100
# RNN Size
rnn_size = 528
# Embedding Dimension Size
embed_dim = 500
# Sequence Length
seq_length = 400
# Learning Rate
learning_rate = 0.005
# Show stats for every n number of batches
show_every_n_batches = 10

save_dir = './save'

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

save_params((seq_length, save_dir))

import tensorflow as tf
import numpy as np

_, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
seq_length, load_dir = load_params()

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    inputs = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return inputs, initial_state, final_state, probs

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    idx = np.argmax(probabilities)
    return int_to_vocab[idx]


gen_length = 100
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'is'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)



