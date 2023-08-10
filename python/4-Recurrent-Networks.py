from __future__ import print_function

import theano
from theano import tensor as T
import numpy as np

np.random.seed(42)
rng = np.random.RandomState(1337)
dtype = theano.config.floatX

def get_weights(n_in, n_out):
    mag = 4. * np.sqrt(6. / (n_in + n_out))
    W_value = np.asarray(rng.uniform(low=-mag, high=mag, size=(n_in, n_out)), dtype=dtype)
    W = theano.shared(value=W_value, name='W_%d_%d' % (n_in, n_out), borrow=True, strict=False)
    return W

def get_bias(n_out):
    b_value = np.asarray(np.zeros((n_out,), dtype=dtype), dtype=theano.config.floatX)
    b = theano.shared(value=b_value, name='b_%d' % n_out, borrow=True, strict=False)
    return b

def rmsprop(cost, params, learning_rate, rho=0.9, epsilon=1e-6):
    updates = list()
    for param in params:
        accu = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=dtype),
                             broadcastable=param.broadcastable)
        grad = T.grad(cost, param)
        accu_new = rho * accu + (1 - rho) * grad ** 2

        updates.append((accu, accu_new))
        updates.append((param, param - (learning_rate * grad / T.sqrt(accu_new + epsilon))))
    return updates

def test(n_in, n_out, X, output, params):
    output = output[-1, :] # get the last timestep from the network
    y = T.matrix(name='y', dtype=dtype) # the target variable
    lr = T.scalar(name='lr', dtype=dtype) # the learning rate (as a variable we can change)

    # minimize binary crossentropy
    xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    cost = xent.mean()
    
    # use rmsprop to get the network updates
    updates = rmsprop(cost, params, lr)

    # generate our testing data
    t_sets = 10
    X_datas = [np.asarray(rng.rand(20, n_in) > 0.5, dtype=dtype) for _ in range(t_sets)]
    y_datas = [np.asarray(rng.rand(1, n_out) > 0.5, dtype=dtype) for _ in range(t_sets)]

    # theano functions for training and testing
    train = theano.function([X, y, lr], [cost], updates=updates)
    test = theano.function([X], [output])

    # some starting parameters
    l = 0.1
    n_train = 1000

    # calculate and display the cost
    cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
    print('Before training:', cost)

    for i in range(n_train):
        for X_data, y_data in zip(X_datas, y_datas):
            train(X_data, y_data, l)

        if (i+1) % (n_train / 5) == 0:
            cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
            print('%d (lr = %f):' % (i+1, l), cost)
            l *= 0.5

def generate_and_test_vanilla_rnn(n_in, n_hidden, n_out):
    X = T.matrix(name='X', dtype=dtype)

    # the weights being used in the network
    w_in = get_weights(n_in, n_hidden)
    w_hidden = get_weights(n_hidden, n_hidden)
    w_out = get_weights(n_hidden, n_out)

    # the biases
    b_hidden = get_bias(n_hidden)
    b_out = get_bias(n_out)
    h_0 = get_bias(n_hidden)
    
    # collect all the parameters here so we can pass them to the optimizer
    params = [w_in, b_hidden, w_out, b_out, w_hidden, h_0]
    
    # define the recurrence here
    def step(x_t, h_tm1):
        h_t = T.tanh(T.dot(x_t, w_in) + T.dot(h_tm1, w_hidden) + b_hidden)
        y_t = T.nnet.sigmoid(T.dot(h_t, w_out) + b_out)
        return h_t, y_t

    [_, output], _ = theano.scan(fn=step, sequences=X, outputs_info=[h_0, None], n_steps=X.shape[0])

    test(n_in, n_out, X, output, params)

generate_and_test_vanilla_rnn(10, 50, 1)

def generate_and_test_lstm(n_in, n_hidden, n_out):
    X = T.matrix(name='X', dtype=dtype)

    # there are a lot of parameters, so let's add them incrementally
    params = list()

    # input gate
    w_in_input = get_weights(n_in, n_hidden)
    w_hidden_input = get_weights(n_hidden, n_hidden)
    b_input = get_bias(n_hidden)
    params += [w_in_input, w_hidden_input, b_input]

    # forget gate
    w_in_forget = get_weights(n_in, n_hidden)
    w_hidden_forget = get_weights(n_hidden, n_hidden)
    b_forget = get_bias(n_hidden)
    params += [w_in_forget, w_hidden_forget, b_forget]

    # output gate
    w_in_output = get_weights(n_in, n_hidden)
    w_hidden_output = get_weights(n_hidden, n_hidden)
    b_output = get_bias(n_hidden)
    params += [w_in_output, w_hidden_output, b_output]

    # hidden state
    w_in_hidden = get_weights(n_in, n_hidden)
    w_hidden_hidden = get_weights(n_hidden, n_hidden)
    b_hidden = get_bias(n_hidden)
    params += [w_in_hidden, w_hidden_hidden, b_hidden]

    # output
    w_out = get_weights(n_hidden, n_out)
    b_out = get_bias(n_out)
    params += [w_out, b_out]

    # starting hidden and memory unit state
    h_0 = get_bias(n_hidden)
    c_0 = get_bias(n_hidden)
    params += [h_0, c_0]
    
    # define the recurrence here
    def step(x_t, h_tm1, c_tm1):
        input_gate = T.nnet.sigmoid(T.dot(x_t, w_in_input) + T.dot(h_tm1, w_hidden_input) + b_input)
        forget_gate = T.nnet.sigmoid(T.dot(x_t, w_in_forget) + T.dot(h_tm1, w_hidden_forget) + b_forget)
        output_gate = T.nnet.sigmoid(T.dot(x_t, w_in_output) + T.dot(h_tm1, w_hidden_output) + b_output)

        candidate_state = T.tanh(T.dot(x_t, w_in_hidden) + T.dot(h_tm1, w_hidden_hidden) + b_hidden)
        memory_unit = c_tm1 * forget_gate + candidate_state * input_gate

        h_t = T.tanh(memory_unit) * output_gate
        y_t = T.nnet.sigmoid(T.dot(h_t, w_out) + b_out)

        return h_t, memory_unit, y_t

    [_, _, output], _ = theano.scan(fn=step, sequences=X, outputs_info=[h_0, c_0, None], n_steps=X.shape[0])

    test(n_in, n_out, X, output, params)

generate_and_test_lstm(10, 50, 1)

def generate_and_test_gru(n_in, n_hidden, n_out):
    X = T.matrix(name='X', dtype=dtype)

    # there are a lot of parameters, so let's add them incrementally
    params = list()

    # update gate
    w_in_update = get_weights(n_in, n_hidden)
    w_hidden_update = get_weights(n_hidden, n_hidden)
    b_update = get_bias(n_hidden)
    params += [w_in_update, w_hidden_update, b_update]

    # reset gate
    w_in_reset = get_weights(n_in, n_hidden)
    w_hidden_reset = get_weights(n_hidden, n_hidden)
    b_reset = get_bias(n_hidden)
    params += [w_in_reset, w_hidden_reset, b_reset]

    # hidden layer
    w_in_hidden = get_weights(n_in, n_hidden)
    w_reset_hidden = get_weights(n_hidden, n_hidden)
    b_in_hidden = get_bias(n_hidden)
    params += [w_in_hidden, w_reset_hidden, b_in_hidden]

    # output
    w_out = get_weights(n_hidden, n_out)
    b_out = get_bias(n_out)
    params += [w_out, b_out]

    # starting hidden state
    h_0 = get_bias(n_hidden)
    params += [h_0]
    
    # define the recurrence here
    def step(x_t, h_tm1):
        update_gate = T.nnet.sigmoid(T.dot(x_t, w_in_update) + T.dot(h_tm1, w_hidden_update) + b_update)
        reset_gate = T.nnet.sigmoid(T.dot(x_t, w_in_reset) + T.dot(h_tm1, w_hidden_reset) + b_reset)
        h_t_temp = T.tanh(T.dot(x_t, w_in_hidden) + T.dot(h_tm1 * reset_gate, w_reset_hidden) + b_in_hidden)

        h_t = (1 - update_gate) * h_t_temp + update_gate * h_tm1
        y_t = T.nnet.sigmoid(T.dot(h_t, w_out) + b_out)

        return h_t, y_t

    [_, output], _ = theano.scan(fn=step, sequences=X, outputs_info=[h_0, None], n_steps=X.shape[0])

    test(n_in, n_out, X, output, params)

generate_and_test_gru(10, 50, 1)

# parameters
input_dims, output_dims = 10, 2
sequence_length = 20
n_test = 10

# generate some random data to train on
X_data = np.asarray([np.asarray(rng.rand(20, input_dims) > 0.5, dtype=dtype) for _ in range(n_test)])
y_data = np.asarray([np.asarray(rng.rand(output_dims,) > 0.5, dtype=dtype) for _ in range(n_test)])

# put together rnn models
from keras.layers import Input
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop
from keras.models import Model

input_sequence = Input(shape=(sequence_length, input_dims,), dtype='float32')

# this is so much easier!
vanilla = SimpleRNN(output_dims, return_sequences=False)
lstm = LSTM(output_dims, return_sequences=False)
gru = GRU(output_dims, return_sequences=False)
rnns = [vanilla, lstm, gru]

# train the models
for rnn in rnns:
    model = Model(input=[input_sequence], output=rnn(input_sequence))
    model.compile(optimizer=RMSprop(lr=0.1), loss='binary_crossentropy')
    print('-- %s --' % rnn.__class__.__name__)
    print('Error before: {}'.format(model.evaluate([X_data], [y_data], verbose=0)))
    model.fit([X_data], [y_data], nb_epoch=1000, verbose=0)
    print('Error after: {}'.format(model.evaluate([X_data], [y_data], verbose=0)))

