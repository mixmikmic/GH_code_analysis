import helper
import numpy as np
from collections import Counter
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import re

data_dir = 'aggregate_lyrics.txt'
text = helper.load_data(data_dir)

view_sentence_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of lyrics split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    word_counts = set(text)
    int_to_vocab = {ii: word for ii, word in enumerate(word_counts)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}    
    
    return (vocab_to_int, int_to_vocab)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokens = {'.':'||Period||',
              ',':'||Comma||',
              '"':'||Quotation_Mark||',
              ';':'||Semicolon||',
              '!':'||Exclamation_Mark||',
              '؟':'||Question_Mark||',
              '(':'||Left_Parantheses||',
              ')':'||Right_Parantheses||',
              '--':'||Dash||',
              '\n':'||Return||'
             }
    return tokens

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

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
    inputs         = tf.placeholder(tf.int32,[None, None], name='input')
    targets        = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate  = tf.placeholder(tf.float32, name='learning_rate')
    return (inputs, targets, learning_rate)

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    Cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
    InitialState = Cell.zero_state(batch_size, tf.float32)
    InitialState = tf.identity(InitialState, name='initial_state')
    
    return (Cell, InitialState)

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1), dtype=tf.float32)
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    return embed

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """

    Outputs, FinalState = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) 
    FinalState =  tf.identity(FinalState, name='final_state')
    return (Outputs, FinalState)

def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    
    embedded = get_embed(input_data=input_data, vocab_size=vocab_size, embed_dim=200)
    outputs, FinalState = build_rnn(cell=cell, inputs=embedded)
    batch_size, embed_size = input_data.get_shape()
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return (logits, FinalState)

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text)//(batch_size*seq_length)

    valid_text = int_text[:n_batches*batch_size*seq_length+1]
    
    result = np.ndarray((n_batches,2,batch_size,seq_length), dtype=int)
    step = n_batches*seq_length    
    
    #print(valid_text)
    
    for batch in range(n_batches):
        batch_walk = batch*seq_length
        x = []
        y = []
        for binn in range(batch_size):
            idx = batch_walk + binn * step    # start from this index
            result[batch][0][binn] = valid_text[idx   : idx    +seq_length]
            result[batch][1][binn] = valid_text[idx+1 : idx+1  +seq_length]   

    return result

num_epochs = 10
batch_size = 128
rnn_size = 256
seq_length = 50
learning_rate = 0.01
show_every_n_batches = 10
save_dir = './save'

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

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
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
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

helper.save_params((seq_length, save_dir))

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

list(vocab_to_int.items())[:10]

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name(name='input:0')
    InitialStateTensor = loaded_graph.get_tensor_by_name(name='initial_state:0')
    FinalStateTensor = loaded_graph.get_tensor_by_name(name='final_state:0')
    ProbsTensor = loaded_graph.get_tensor_by_name(name='probs:0')
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)

def pick_word(preds, int_to_vocab, top_n=10):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(list(int_to_vocab.values()), 1, p=p)[0]
    
    return c

gen_length = 60
prime_word = u'عمري'

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
        if pred_word == gen_sentences[len(gen_sentences)-1]:
            continue
        gen_sentences.append(pred_word)
    
    # Remove tokens
    lyrics = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        lyrics = lyrics.replace(' ' + token, key)
    lyrics = re.sub('(\n){2,}', '\n', lyrics)
        
    print(lyrics.strip())



