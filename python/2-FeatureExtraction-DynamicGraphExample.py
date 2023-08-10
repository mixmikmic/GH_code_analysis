# boilerplate
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
import tensorflow_fold as td

RNN_FEATURE_SIZE = 4
A_SIZE = 3
B_SIZE = 5

input_sequence = [
    {'type': 'A', 'data': [1, 2, 3]},
    {'type': 'B', 'data': [5, 4, 3, 2, 1]},
    {'type': 'A', 'data': [3, 2, 1]},
]

input_sequence

feature_from_A = td.Vector(A_SIZE) >> td.FC(RNN_FEATURE_SIZE)
feature_from_B = td.Vector(B_SIZE) >> td.FC(RNN_FEATURE_SIZE)

feature_from_A.eval(input_sequence[0]['data'])

#feature_from_A.eval(input_sequence[1]['data'])  # fails since it gets the wrong size of input

feature = td.OneOf(
    key_fn=lambda x: x['type'],
    case_blocks={
        'A': td.GetItem('data') >> feature_from_A,
        'B': td.GetItem('data') >> feature_from_B
    }
)

[feature.eval(input_) for input_  in input_sequence]

feature_sequence = td.Map(feature)

feature_sequence.eval(input_sequence)

lstm_cell = td.ScopedLayer(
    tf.contrib.rnn.BasicLSTMCell(num_units=16),
    'lstm_cell'
)
lstm_output = feature_sequence >> td.RNN(lstm_cell, name='lstm')

lstm_output.eval(input_sequence)
# Format:
# (
#    [state_0, ... state_T],
#    (cell_T, state_T)
# )

