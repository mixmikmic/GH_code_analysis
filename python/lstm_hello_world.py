get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rc('image', interpolation='nearest', cmap='gray')
mpl.rc('figure', figsize=(20,10))

X = np.array([[[1],[1],[0]], [[1],[0],[1]], [[0],[1],[1]]])
y = np.array([[1], [1], [0]])

# X = np.array([[[1],[0],[0]], [[0],[1],[0]], [[0],[0],[1]]])
# y = np.array([[1], [0], [0]])

# input: 3 samples of 3-step sequences with 1 feature
# input: 3 samples with 1 feature
X.shape, y.shape

from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM

# model = Sequential()
# # return_sequences=False
# model.add(LSTM(output_dim=1, input_shape=(3, 1)))
# # since the LSTM layer has only one output after activation we can directly use as model output
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

# This models is probably too easy and it is not able to overfit on the training dataset.
# For LSTM output dim 3 it works ok (after a few hundred epochs).

model = Sequential()
model.add(LSTM(output_dim=3, input_shape=(3, 1)))
# Since the LSTM layer has multiple outputs and model has single one
# we need to add another Dense layer with single output.
# In case the LSTM would return sequences we would use TimeDistributedDense layer.
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

model.count_params()

model.fit(X, y, nb_epoch=500, show_accuracy=True)

plt.plot(model.predict_proba(X).flatten(), 'rx')
plt.plot(model.predict_classes(X).flatten(), 'ro')
plt.plot(y.flatten(), 'g.')
plt.xlim(-0.1, 2.1)
plt.ylim(-0.1, 1.1)

model.predict_proba(X)

model.predict_classes(X)

# del model

weight_names = ['W_i', 'U_i', 'b_i',
 'W_c', 'U_c', 'b_c',
 'W_f', 'U_f', 'b_f',
 'W_o', 'U_o', 'b_o']

weight_shapes = [w.shape for w in model.get_weights()]
# for n, w in zip(weight_names, weight_shapes):
#     print(n, ':', w)
print(weight_shapes)

def pad_vector_shape(s):
    return (s[0], 1) if len(s) == 1 else s

all_shapes = np.array([pad_vector_shape(s) for s in weight_shapes])
all_shapes

for w in model.get_weights():
    print(w)

all_weights = np.zeros((all_shapes[:,0].sum(axis=0), all_shapes[:,1].max(axis=0)))

def add_weights(src, target):
    target[0] = src[0]
    target[1:4] = src[1]
    target[4:7,0] = src[2]
    
for i in range(4):
    add_weights(model.get_weights()[i*3:(i+1)*3], all_weights[i*7:(i+1)*7])

all_weights[28:31,0] = model.get_weights()[12].T
all_weights[31,0] = model.get_weights()[13]
    
plt.imshow(all_weights.T)


from matplotlib.patches import Rectangle

ax = plt.gca()
ax.add_patch(Rectangle([-.4, -0.4], 28-0.2, 3-0.2, fc='none', ec='r', lw=2, alpha=0.75))
ax.add_patch(Rectangle([28 - .4, -0.4], 3-0.2, 3-0.2, fc='none', ec='g', lw=2, alpha=0.75))
ax.add_patch(Rectangle([31 - .4, -0.4], 1-0.2, 3-0.2, fc='none', ec='b', lw=2, alpha=0.75))

plt.savefig('weights_110.png')



