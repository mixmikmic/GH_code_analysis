import numpy as np

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks.recurrent import SimpleRecurrent

x = tensor.tensor3('x')

rnn = SimpleRecurrent(dim=3, activation=Identity(), weights_init=initialization.Identity())
rnn.initialize()
h = rnn.apply(x)

f = theano.function([x], h)
print(f(np.ones((3, 1, 3), dtype=theano.config.floatX))) 

from blocks.bricks import Linear

doubler = Linear(
             input_dim=3, output_dim=3, weights_init=initialization.Identity(2),
             biases_init=initialization.Constant(0))
doubler.initialize()
h_doubler = rnn.apply(doubler.apply(x))

f = theano.function([x], h_doubler)
print(f(np.ones((3, 1, 3), dtype=theano.config.floatX))) 

h0 = tensor.matrix('h0')
h = rnn.apply(inputs=x, states=h0)

f = theano.function([x, h0], h)
print(f(np.ones((3, 1, 3), dtype=theano.config.floatX),
        np.ones((1, 3), dtype=theano.config.floatX))) 

ignore="""
digraph feedback_rnn {
node [shape=plaintext,label="(1, 1, 1)"];
x_1; x_2; x_3;

node [shape=plaintext];
h1_0 [label="(0, 0, 0)"]; h1_1 [label="(1, 1, 1)"];
h1_2 [label="(4, 4, 4)"]; h1_3 [label="(12, 12, 12)"];
h2_0 [label="(0, 0, 0)"]; h2_1 [label="(1, 1, 1)"];
h2_2 [label="(3, 3, 3)"]; h2_3 [label="(8, 8, 8)"];

node [shape=diamond,regular=1,label="+"];
plus_1_1; plus_1_2; plus_1_3; plus_2_1; plus_2_2; plus_2_3;

x_1 -> plus_1_1; x_2 -> plus_1_2; x_3 -> plus_1_3;
h1_0 -> plus_1_1 -> h1_1 -> plus_1_2 -> h1_2 -> plus_1_3 -> h1_3;
h2_0 -> plus_2_1 -> h2_1 -> plus_2_2 -> h2_2 -> plus_2_3 -> h2_3;
h2_0 -> plus_1_1; h2_1 -> plus_1_2; h2_2 -> plus_1_3;

edge [style=invis];
h2_0 -> h1_0; h2_1 -> h1_1; h2_2 -> h1_2; h2_3 -> h1_3;
plus_2_1 -> plus_1_1; plus_2_2 -> plus_1_2; plus_2_3 -> plus_1_3;

{ rank=source; h2_0, h2_1, h2_2, h2_3, plus_2_1, plus_2_2, plus_2_3 }
{ rank=same; h1_0, h1_1, h1_2, h1_3, plus_1_1, plus_1_2, plus_1_3 }
{ rank=sink; x_1, x_2, x_3}
}
"""

from blocks.bricks.recurrent import BaseRecurrent, recurrent
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

