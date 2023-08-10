import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
import numpy as np

class seq2seq_example:

    # Constants
    tokens         = {"PAD": 0, "EOS": 1, "GO": 2, "UNK": 3}
    minLength      = 5
    maxLength      = 10
    samples        = 10000
    vocab_size     = 50
    embedding_size = 15
    dropout        = 0.3
    layers         = 2
    layer_size     = 100
    batch_size     = 50
    beam_width     = 4

    def __init__(self):
        
        # Random integers up to the vocab_size (not including reserved integers)
        self.data = np.random.randint(
            low  = len(self.tokens),
            high = self.vocab_size,
            size = (self.samples, self.maxLength))
        
        # Assign a random length to each sequence from minLength to maxLength
        self.dataLens = np.random.randint(
            low  = self.minLength,
            high = self.maxLength,
            size = self.samples)
        
        # Create labels by sorting the original data
        self.dataLabels = np.ones_like(self.data) * self.tokens['PAD']
        for i in range(len(self.data)):
            self.data[i, self.dataLens[i]:] = self.tokens['PAD']
            self.dataLabels[i, :self.dataLens[i]] = np.sort(self.data[i, :self.dataLens[i]])
       
        # Make placeholders and stuff
        self.make_inputs()

        # Build the compute graph
        self.build_graph()

    # Create the inputs to the graph (placeholders and stuff)
    def make_inputs(self):
        self.input     = tf.placeholder(tf.int32, (self.batch_size, self.maxLength))
        self.lengths   = tf.placeholder(tf.int32, (self.batch_size,))
        self.labels    = tf.placeholder(tf.int32, (self.batch_size, self.maxLength))
        self.keep_prob = tf.placeholder(tf.float32)

        # Embed encoder input
        self.enc_input = tf.contrib.layers.embed_sequence(
            ids        = self.input,
            vocab_size = self.vocab_size,
            embed_dim  = self.embedding_size)

        # Decoder input (GO + label + EOS)
        eos = tf.one_hot(
            indices  = self.lengths,
            depth    = self.maxLength,
            on_value = self.tokens['EOS'])
        
        self.add_eos = self.labels + eos
        go_tokens = tf.constant(self.tokens['GO'], shape=[self.batch_size, 1])
        pre_embed_dec_input = tf.concat((go_tokens, self.add_eos), 1)
        
        # Embed decoder input
        self.dec_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))
        self.dec_input = tf.nn.embedding_lookup(self.dec_embed, pre_embed_dec_input)

    def one_layer_cell(self):
        return rnn.DropoutWrapper(rnn.LSTMCell(self.layer_size), self.keep_prob)
    
    def cell(self):
        return rnn.MultiRNNCell([self.one_layer_cell() for _ in range(self.layers)])
    
    def decoder_cell(self, inputs, lengths):
        attention_mechanism = seq2seq.LuongAttention(
                num_units              = self.layer_size,
                memory                 = inputs,
                memory_sequence_length = lengths,
                scale                  = True)

        return seq2seq.AttentionWrapper(
                cell                 = self.cell(),
                attention_mechanism  = attention_mechanism,
                attention_layer_size = self.layer_size)
    
    # Build the compute graph. First encoder, then decoder, then train/test ops
    def build_graph(self):
        
        # Build the encoder
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell            = self.cell(),
            inputs          = self.enc_input,
            sequence_length = self.lengths,
            dtype           = tf.float32)

        # Replicate the top-most encoder state for starting state of all layers in the decoder
        dec_start_state = tuple(enc_state[-1] for _ in range(self.layers))
        
        # Output layer converts from layer size to vocab size
        output = Dense(self.vocab_size,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1))
        
        # Training decoder: scheduled sampling et al.
        with tf.variable_scope("decode"):
            
            cell = self.decoder_cell(enc_outputs, self.lengths)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            init_state = init_state.clone(cell_state=dec_start_state)
            
            train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs               = self.dec_input,
                    sequence_length      = self.lengths,
                    embedding            = self.dec_embed,
                    sampling_probability = 0.1)

            train_decoder = seq2seq.BasicDecoder(
                    cell          = cell,
                    helper        = train_helper,
                    initial_state = init_state,
                    output_layer  = output)
            
            train_output, _, train_lengths = seq2seq.dynamic_decode(
                    decoder            = train_decoder,
                    maximum_iterations = self.maxLength)
        
        # Tile inputs for beam search decoder
        dec_start_state = seq2seq.tile_batch(dec_start_state, self.beam_width)
        enc_outputs = seq2seq.tile_batch(enc_outputs, self.beam_width)
        lengths = seq2seq.tile_batch(self.lengths, self.beam_width)
        
        # Share weights with training decoder
        with tf.variable_scope("decode", reuse=True):
            
            cell = self.decoder_cell(enc_outputs, lengths)
            init_state = cell.zero_state(self.batch_size * self.beam_width, tf.float32)
            init_state = init_state.clone(cell_state=dec_start_state)
            
            test_decoder = seq2seq.BeamSearchDecoder(
                    cell          = cell,
                    embedding     = self.dec_embed,
                    start_tokens  = tf.ones_like(self.lengths) * self.tokens['GO'],
                    end_token     = self.tokens['EOS'],
                    initial_state = init_state,
                    beam_width    = self.beam_width,
                    output_layer  = output)
            
            test_output, _, test_lengths = seq2seq.dynamic_decode(
                    decoder            = test_decoder,
                    maximum_iterations = self.maxLength)
        
        # Create train op. Add one to train lengths, to include EOS
        mask = tf.sequence_mask(train_lengths + 1, self.maxLength - 1, dtype=tf.float32)
        self.cost = seq2seq.sequence_loss(train_output.rnn_output, self.add_eos[:, :-1], mask)
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        # Create test error rate op. Remove one from lengths to exclude EOS
        predicts = self.to_sparse(test_output.predicted_ids[:,:,0], test_lengths[:, 0] - 1)
        labels = self.to_sparse(self.labels, self.lengths)
        self.error_rate = tf.reduce_mean(tf.edit_distance(predicts, labels))

    # Convert a dense matrix into a sparse matrix (for e.g. edit_distance)
    def to_sparse(self, tensor, lengths):
        mask = tf.sequence_mask(lengths, self.maxLength)
        indices = tf.to_int64(tf.where(tf.equal(mask, True)))
        values = tf.to_int32(tf.boolean_mask(tensor, mask))
        shape = tf.to_int64(tf.shape(tensor))
        return tf.SparseTensor(indices, values, shape)

    # Divide training samples into batches
    def batchify(self):

        for i in range(self.samples // self.batch_size):
            yield self.next_batch(i)

    # Create a single batch at i * batch_size
    def next_batch(self, i):

        start = i * self.batch_size
        stop = (i+1) * self.batch_size

        batch = {
                self.input:     self.data[start:stop],
                self.lengths:   self.dataLens[start:stop],
                self.labels:    self.dataLabels[start:stop],
                self.keep_prob: 1. - self.dropout
        }

        return batch

    # Create a random test batch
    def test_batch(self):

        data = np.random.randint(
            low  = len(self.tokens),
            high = self.vocab_size,
            size = (self.batch_size, self.maxLength))
        
        dataLens = np.random.randint(
            low  = self.minLength,
            high = self.maxLength,
            size = self.batch_size)
        
        dataLabels = np.zeros_like(data)
        for i in range(len(data)):
            data[i, dataLens[i]:] = self.tokens['PAD']
            dataLabels[i, :dataLens[i]] = np.sort(data[i, :dataLens[i]])

        return {
                self.input: data,
                self.lengths: dataLens,
                self.labels: dataLabels,
                self.keep_prob: 1.
        }

s2s = seq2seq_example()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(50):
        
        # Keep track of average train cost for this epoch
        train_cost = 0
        for batch in s2s.batchify():
            train_cost += sess.run([s2s.train_op, s2s.cost], batch)[1]
        train_cost /= s2s.samples / s2s.batch_size
        
        # Test time
        er = sess.run(s2s.error_rate, s2s.test_batch())
        
        print("Epoch", (epoch + 1), "train loss:", train_cost, "test error:", er)

