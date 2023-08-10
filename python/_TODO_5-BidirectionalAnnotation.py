import numpy as np

import theano
from theano import tensor

#from blocks import initialization
from blocks.bricks import Identity, Linear, Tanh, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional, BaseRecurrent
from blocks.bricks.parallel import Merge
#from blocks.bricks.parallel import Fork

from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, WEIGHT, OUTPUT

vocab_size=3
embedding_dim=3
labels_size=10

lookup = LookupTable(vocab_size, embedding_dim)

encoder = Bidirectional(SimpleRecurrent(dim=embedding_dim, activation=Tanh()))

mlp = MLP([Softmax()], [embedding_dim, labels_size],
          weights_init=IsotropicGaussian(0.01),
          biases_init=Constant(0))

#encoder.prototype.apply.sequences
#dir(encoder.prototype.apply.sequences)

#combine = Merge(input_dims=dict(), output_dim=labels_size)
#labelled = Softmax( encoder )


x = tensor.imatrix('features')
y = tensor.imatrix('targets')

probs = mlp.apply(encoder.apply(lookup.apply(x)))

cost = CategoricalCrossEntropy().apply(y.flatten(), probs)

#cg = ComputationGraph([cost])
cg = ComputationGraph([probs])
cg.variables
#VariableFilter(roles=[OUTPUT])(cg.variables)

#dir(cg.outputs)
#np.shape(cg.outputs)

#mlp = MLP([Softmax()], [embedding_dim*2, labels_size],
#          weights_init=IsotropicGaussian(0.01),
#          biases_init=Constant(0))
#mlp.initialize()

#fork = Fork([name for name in encoder.prototype.apply.sequences if name != 'mask'])
#fork.input_dim = dimension
#fork.output_dims = [dimension for name in fork.input_names]
#print(fork.output_dims)

#readout = Readout(
#    readout_dim=labels_size,
#    source_names=[encodetransition.apply.states[0], attention.take_glimpses.outputs[0]],
#    emitter=SoftmaxEmitter(name="emitter"),
#    #feedback_brick=LookupFeedback(alphabet_size, dimension),
#    name="readout")

h0 = tensor.matrix('h0')
h = rnn.apply(inputs=x, states=h0)

f = theano.function([x, h0], h)
print(f(np.ones((3, 1, 3), dtype=theano.config.floatX),
        np.ones((1, 3), dtype=theano.config.floatX))) 



class FeedbackRNN(BaseRecurrent):
    def __init__(self, dim, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        self.dim = dim
        self.first_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='first_recurrent_layer',
            weights_init=initialization.Identity())
        self.second_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='second_recurrent_layer',
            weights_init=initialization.Identity())
        self.children = [self.first_recurrent_layer,
                         self.second_recurrent_layer]

    @recurrent(sequences=['inputs'], contexts=[],
               states=['first_states', 'second_states'],
               outputs=['first_states', 'second_states'])
    def apply(self, inputs, first_states=None, second_states=None):
        first_h = self.first_recurrent_layer.apply(
            inputs=inputs, states=first_states + second_states, iterate=False)
        second_h = self.second_recurrent_layer.apply(
            inputs=first_h, states=second_states, iterate=False)
        return first_h, second_h

    def get_dim(self, name):
        return (self.dim if name in ('inputs', 'first_states', 'second_states')
                else super(FeedbackRNN, self).get_dim(name))

x = tensor.tensor3('x')

feedback = FeedbackRNN(dim=3)
feedback.initialize()
first_h, second_h = feedback.apply(inputs=x)

f = theano.function([x], [first_h, second_h])
for states in f(np.ones((3, 1, 3), dtype=theano.config.floatX)):
    print(states) 

