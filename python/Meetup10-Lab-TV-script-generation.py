import helper2

data_dir = './data/moes_tavern_lines.txt'
text = helper2.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]

view_sentence_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

import numpy as np

def create_lookup_tables(text):
    """
    Creates lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    d = {}
    d['.'] =  '||PERIOD||'
    d[','] =  '||COMMA||'
    d['"'] =  '||QUOTATION_MARK||'
    d[';'] =  '||SEMICOLON||'
    d['!'] =  '||EXCLAMATION_MARK||'
    d['?'] =  '||QUESTION_MARK||'
    d['('] =  '||LEFT_PAREN||'
    d[')'] =  '||RIGHT_PAREN||'
    d['--'] =  '||DASH||'
    d['?'] =  '||QUESTION_MARK>'
    d['\n'] =  '||NEW_LINE||'

    return d

# Preprocess Training, Validation, and Testing Data
helper2.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

import helper2
import numpy as np

int_text, vocab_to_int, int_to_vocab, token_dict = helper2.load_preprocess()

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
    Creates TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    
    # Keep probability placeholder for drop out layers
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
     
    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    """
    Creates an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """       
    # Use a basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.75)
    
    # Stack up multiple LSTM layers, for deep learning
    lstm_layers = 1
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, 'initial_state')
    
    
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Creates embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """

    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    return embed

def build_rnn(cell, inputs):
    """
    Creates a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    
    return outputs, final_state

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
    outputs, FinalState = build_rnn(cell, embed)
    Logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return Logits, FinalState

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = int(len(int_text) / (batch_size * seq_length))

    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.roll(xdata,-1)

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 600
# Embedding Dimension Size
embed_dim = 250
# Sequence Length
seq_length = 64
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 64

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

# Save parameters for checkpoint
helper2.save_params((seq_length, save_dir))

import tensorflow as tf
import numpy as np
import helper2

_, vocab_to_int, int_to_vocab, token_dict = helper2.load_preprocess()
seq_length, load_dir = helper2.load_params()

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
    Picks the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return int_to_vocab[np.argmax(probabilities)]
    #return int_to_vocab[np.random.choice(len(int_to_vocab), p=probabilities)]

gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
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
         
        pred_word = pick_word(probabilities[0,dyn_seq_length-1,:], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)

