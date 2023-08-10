def generate_data(p, batch_size=BATCH_SIZE, data=in_text, return_target=True):
    '''
    Produces a semi-redundant batch of training samples from the location 'p' in the provided string (data).

    Inputs: 
        p (int): Position in the input text to start at
        batch_size (int): The number of frame shifts to make/training samples to make
        data (str): The input corpus
        return_target (boolean): Flag to determine whether to return the target vector

    Returns:
        x (np array): Input data matrix that takes the shape of (batch_size x SEQ_LENGTH x vocab_size)
        y (np array): Output data representing one-hot encoding of guessed character
    '''

    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
        if(return_target):
            y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y,dtype='int32')

# Step 1 -> Build the input layer of the network. 
# Recurrent layers expect shape (batch size, SEQ_LENGTH, num_features)
l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

# Step 2 -> Build LSTM layer with takes l_in as input layer + clip gradient
l_forward_1 = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

l_forward_2 = lasagne.layers.LSTMLayer(
    l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh)

# Step 3 -> Feed the output from the recurrent LSTM layers into a traditional Dense Layer
# We only care about the final prediction here, so we isolate that and feed it into the next layer
# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

# Step 4 -> Sliced output passed through softmax nonlinearity to create probability distribution of the prediction
# The output of this stage is (batch_size, vocab_size)
l_out = lasagne.layers.DenseLayer(l_forward_slice,
                                  num_units=vocab_size,
                                  W = lasagne.init.Normal(),
                                  nonlinearity=lasagne.nonlinearities.softmax)

# Theano tensor for the targets
target_values = T.ivector('target_output')

# lasagne.layers.get_output produces a variable for the output of the net
network_output = lasagne.layers.get_output(l_out)

# The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out,trainable=True)

# Compute AdaGrad updates for training
logging.info("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

# Theano functions for training and computing cost
logging.info("Compiling functions ...")
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

# Generate probabilities used to generate text from the network given the current state and seed text input
probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

def generate_text(N=200):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        The function works in three steps:
        1. Convert string set in "generation_phrase" (which must be over SEQ_LENGTH characters long) 
           to encoded format. 
        2. Use LSTM to predict the next character and store it in a (dynamic) list sample_ix using the 'probs'
           function which was compiled above.
        3. Construct a new sequence using all but first characters of the provided string and the 
           predicted character. This sequence is then used to generate yet another character.
           
        This process continues for "N" characters. 
        '''

        assert(len(generation_phrase)>=SEQ_LENGTH)
        sample_ix = []
        x,_ = gen_data(len(generation_phrase)-SEQ_LENGTH, 1, generation_phrase,0)

        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x).ravel())
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = generation_phrase + ''.join(ix_to_char[ix] for ix in sample_ix)    
        logging.info("----\n %s \n----" % random_snippet)

for it in xrange(data_size * num_epochs / BATCH_SIZE):
    generate_text() # Generate text using the p^th character as the start. 

    avg_cost = 0;
    for i in range(PRINT_FREQ):
        x,y = gen_data(p)

        p += SEQ_LENGTH + BATCH_SIZE - 1 
        if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
            logging.info('Carriage Return')
            p = 0;

        avg_cost += train(x, y)
    logging.info("Epoch {} average loss = {}".format(i*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))

