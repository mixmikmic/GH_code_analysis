import numpy as np
import mxnet as mx

import random
random.seed(10)

n_samples = 10000
n_numbers = 3 # numbers to operate on
largest = 10

character_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '*', ' ']

input_sequence_length =  8 # (10 + 10 + 10)
output_sequence_length = 4 



def generate_data(n_samples):
    inputs = []
    labels = []
    
    char_to_int = dict((c,i)  for i,c in enumerate(character_set))
    
    for i in range(n_samples):
        lhs = [random.randint(1, largest) for _ in range(n_numbers)]
        op = random.choice(['+', '*'])
        if op == '+':
            rhs = sum(lhs)
        elif op == '*':
            rhs = 1
            for l in lhs:
                rhs *= l
        
        lhs = [str(l) for l in lhs]
        strng = op.join(lhs)
        padded_strng = "%*s" % (input_sequence_length, strng)
        enc_input = [char_to_int[ch] for ch in padded_strng]
        
        #RHS
        padded_strng = "%*s" % (output_sequence_length, rhs)
        enc_lbl = [char_to_int[ch] for ch in padded_strng]
    
        inputs.append(enc_input)
        labels.append(enc_lbl)
        
    return np.array(inputs), np.array(labels)


dataX, dataY = generate_data(n_samples)

print dataX.shape, dataY.shape

# Iterators

batch_size = 32
data_dim = len(character_set)

train_iter = mx.io.NDArrayIter(data=dataX, label=dataY,
                               data_name='data', label_name='target',
                               batch_size=batch_size, shuffle=True)

train_iter.provide_data, train_iter.provide_label

# Lets build the model!

data = mx.sym.var('data')
target = mx.sym.var('target')

# Encoder - Decoder 

lstm1 = mx.rnn.FusedRNNCell(num_hidden=32, prefix="lstm1_", get_next_state=True)
lstm2 = mx.rnn.FusedRNNCell(num_hidden=32, prefix="lstm2_", get_next_state=False)

# convert to one-hot encoding

data_one_hot = mx.sym.one_hot(data, depth=len(character_set))
data_one_hot = mx.sym.transpose(data_one_hot, axes=(1,0,2))

# unroll the loop/lstm

# Note that when unrolling, if 'merge_outputs' is set to True, the 'outputs' is merged into a single symbol
# In the layout, 'N' represents batch size, 'T' represents sequence length, and 'C' represents the
# number of dimensions in hidden states.

l_out, encode_state = lstm1.unroll(length=input_sequence_length, inputs=data_one_hot, layout="TNC")
encode_state_h = mx.sym.broadcast_to(encode_state[0], shape=(output_sequence_length, 0, 0))

# Decoder

decode_out, l2 = lstm2.unroll(length=output_sequence_length, inputs=encode_state_h, layout="TNC")
decode_out = mx.sym.reshape(decode_out, shape=(-1,batch_size))

out = mx.sym.FullyConnected(decode_out, num_hidden=data_dim)
out = mx.sym.reshape(out, shape=(output_sequence_length, -1, data_dim))
out = mx.sym.transpose(out, axes=(1,0,2))

loss = mx.sym.mean(-mx.sym.pick(mx.sym.log_softmax(out), target, axis=-1))
loss  = mx.sym.make_loss(loss)

shape = {"data": (batch_size, dataX[0].shape[0])}
#mx.viz.plot_network(out, shape=shape)

#["cats", "dogs"] ==> [0, 1] ==> [[1, 0], [0, 1]]

# Module


net = mx.mod.Module(symbol=loss,
                   data_names=['data'], label_names=['target'],
                    context=mx.gpu(7)
                   )

net.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
net.init_params(initializer=mx.init.Xavier())
net.init_optimizer(optimizer='adam',
                  optimizer_params={'learning_rate': 1E-3},
                   kvstore=None
                  )

#Train

epochs = 100

total_batches = len(dataX) // batch_size

for epoch in range(epochs):
    avg_loss =0
    train_iter.reset()
    
    for i, data_batch in enumerate(train_iter):
        net.forward_backward(data_batch=data_batch)
        loss = net.get_outputs()[0].asscalar()
        avg_loss += loss
        net.update()
    avg_loss /= total_batches

    print epoch, "%.7f" % avg_loss

# test module
test_net = mx.mod.Module(symbol=out,
                         data_names=['data'],
                         label_names=None,
                         context=mx.gpu(7)) # FusedRNNCell works only with GPU

# data descriptor
data_desc = train_iter.provide_data[0]

# set shared_module = model used for training so as to share same parameters and memory
test_net.bind(data_shapes=[data_desc],
              label_shapes=None,
              for_training=False,
              grad_req='null',
              shared_module=net)

n_test = 100
testX, testY = generate_data(n_test)

testX = np.array(testX, dtype=np.int)

test_net.reshape(data_shapes=[mx.io.DataDesc('data', (1, input_sequence_length))])
predictions = test_net.predict(mx.io.NDArrayIter(testX, batch_size=1)).asnumpy()

print "expression", "predicted", "actual"

correct = 0
for i, prediction in enumerate(predictions):
    x_str = [character_set[j] for j in testX[i]]
    index = np.argmax(prediction, axis=1)
    result = [character_set[j] for j in index]
    label = [character_set[j] for j in testY[i]]
    #print result, label
    if result == label:
        correct +=1
    print "".join(x_str), "".join(result), "    ", "".join(label)
    
print correct, correct/(n_test*1.0)



