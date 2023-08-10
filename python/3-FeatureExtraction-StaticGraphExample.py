# boilerplate
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
#import tensorflow_fold as td

RNN_FEATURE_SIZE = 4
A_SIZE = 3
B_SIZE = 5

input_sequence = [
    {'type': 'A', 'data': [1, 2, 3]},
    {'type': 'B', 'data': [5, 4, 3, 2, 1]},
    {'type': 'A', 'data': [3, 2, 1]},
]

inputs_A = tf.placeholder('float32', shape=(None, None, A_SIZE), name='inputs_A')
inputs_B = tf.placeholder('float32', shape=(None, None, B_SIZE), name='inputs_B')

feature_from_A = tf.layers.dense(inputs_A, RNN_FEATURE_SIZE, activation=tf.nn.relu)
feature_from_B = tf.layers.dense(inputs_B, RNN_FEATURE_SIZE, activation=tf.nn.relu)

sess.run(tf.global_variables_initializer())

sess.run(
    feature_from_A,
    {
        inputs_A: np.array([
            [[1,2,3], [3,2,1]]
        ])  # 3D array - batch of sequences of feature vectors
    }
)

mask_A = tf.placeholder('float32', shape=(None, None, 1), name='mask_A')
mask_B = tf.placeholder('float32', shape=(None, None, 1), name='mask_B')

feature_from_A_masked = feature_from_A * mask_A 
feature_from_B_masked = feature_from_B * mask_B

sess.run(
    feature_from_A_masked,
    {
        inputs_A: np.array([
            [[1,2,3], [np.nan, np.nan, np.nan]]  # makes more sense to pad with zeros though!
        ]),
        mask_A: np.array([
            [[1], [0]]
        ])
    }
)

feature_sequences = feature_from_A_masked + feature_from_B_masked

input_sequences = [
    [
        {'type': 'A', 'data': [1, 2, 3]},
        {'type': 'B', 'data': [5, 4, 3, 2, 1]},
        {'type': 'A', 'data': [3, 2, 1]},
    ],
    [
        {'type': 'B', 'data': [1, 2, 3, 4, 5]},
        {'type': 'B', 'data': [5, 4, 3, 2, 1]},
        {'type': 'A', 'data': [3, 2, 1]},
    ]
]

N = len(input_sequences)
T = len(input_sequences[0])

assert all([len(in_seq) == T for in_seq in input_sequences])
# all input sequences must have same lenght or we need to introduce additional padding


inputs = {
    'A': np.zeros((N, T, A_SIZE)),
    'B': np.zeros((N, T, B_SIZE))
}
masks = {
    'A': np.zeros((N, T, 1)),
    'B': np.zeros((N, T, 1))
}

for n, in_seq in enumerate(input_sequences):
    for t, input_ in enumerate(in_seq): 
        inputs[input_['type']][n, t, :] = np.array(input_['data'])
        masks[input_['type']][n, t, 0] = 1
        
feed_dict = {
    inputs_A: inputs['A'],
    mask_A: masks['A'],
    inputs_B: inputs['B'],
    mask_B: masks['B']
}

for tensor, data in feed_dict.items():
    print(tensor)
    print(data)

sess.run(feature_sequences, feed_dict)



