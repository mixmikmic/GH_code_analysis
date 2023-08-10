get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from time import time

session = tf.Session()

def make_block_linear_operator(block_sizes, variable=tf.Variable,
                               diag_init=0., lowrank_init=0.1):
    dim = sum(block_sizes)
    n_blocks = len(block_sizes)
    psd_args = dict(
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
    )
    diagonal = tf.zeros([dim])
    if hasattr(diag_init, 'shape') or hasattr(diag_init, '__len__'):
        diag_variables = variable(diag_init)
    else:
        diag_variables = variable(diag_init * tf.ones(n_blocks))
    if hasattr(lowrank_init, 'shape') or hasattr(lowrank_init, '__len__'):
        lowrank_variables = variable(lowrank_init)
    else:
        lowrank_variables = variable(lowrank_init * tf.ones(n_blocks))

    offset = 0
    tic = time()
    for i, block_size in enumerate(block_sizes):
        pad_before = offset
        pad_after = dim - (offset + block_size)
        diagonal += tf.pad(tf.exp(diag_variables[i:i+1]) * tf.ones([block_size]),
                           [[pad_before, pad_after]])
        offset += block_size
    linop = tf.linalg.LinearOperatorDiag(diagonal, **psd_args)
    print(f"diagonal operator built in {time() - tic:.3f}s")

    offset = 0
    for i, block_size in enumerate(block_sizes):
        tic = time()
        pad_before = offset
        pad_after = dim - (offset + block_size)
        u = tf.pad(lowrank_variables[i:i+1] * tf.ones([block_size, 1]),
                   [[pad_before, pad_after], [0, 0]])

        # Recursively chain the the LR updates operators: this is the
        # expensive step as it precomputes the solution of A.X = U where
        # A is the previous base operator.
        linop = tf.linalg.LinearOperatorLowRankUpdate(linop, u=u, **psd_args)
        offset += block_size
        print(f"lowrank update operator built in {time() - tic:.3f}s")
    return linop, [diag_variables, lowrank_variables] 

K, variables = make_block_linear_operator(
    [3, 5, 5, 1], diag_init=0., lowrank_init=0.1)

session.run(tf.global_variables_initializer())
get_ipython().run_line_magic('time', 'session.run(K.log_abs_determinant())')

get_ipython().run_line_magic('time', 'session.run(tf.gradients(K.log_abs_determinant(), variables))')

# val_and_grad_fn = tfe.value_and_gradients_function(lambda K: K.log_abs_determinant())
# val_and_grad_fn(K)

K_tensor = K.to_dense()

def matshow(a, cmap=plt.cm.RdBu, figsize=(6, 6)):
    vmax = np.abs(a).max()
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(a, vmin=-vmax, vmax=vmax, cmap=cmap)

matshow(session.run(K_tensor))

session.run(tf.global_variables_initializer())
diag_variables, offdiag_variables = variables
session.run(diag_variables.assign([1, -.1, 0.8, 0]))
session.run(offdiag_variables.assign([0.9, 0.4, 0.7, 1]))
matshow(session.run(K.to_dense()))

np.mean(session.run(K.to_dense()) == 0)

K_expm = tf.linalg.expm(K_tensor)
K_expm_value, K_expm_det = session.run([K_expm, tf.linalg.det(K_expm)])
matshow(K_expm_value)
K_expm_det

matshow(session.run(tf.linalg.inv(K_tensor)));

K_inv = session.run(tf.linalg.inv(K_tensor))
plt.hist(K_inv.ravel(), bins=30)
np.mean(K_inv == 0)

get_ipython().run_line_magic('time', 'big_K, big_variables = make_block_linear_operator([4096] * 8)')

# %load_ext line_profiler

# %lprun -f tf.linalg.LinearOperatorLowRankUpdate.__init__ make_block_linear_operator([4096] * 10)

big_K.shape

np.product(big_K.shape).value / 1e9

sum(np.prod(v.shape).value for v in big_variables)

get_ipython().run_line_magic('time', 'big_det = big_K.log_abs_determinant()')

session.run(tf.global_variables_initializer())
get_ipython().run_line_magic('time', 'session.run(big_det)')

block_sizes = [100, 80, 120, 5]
# ground truth distribution parameters:
K_gt, variables_gt = make_block_linear_operator(
    block_sizes,
    diag_init=[0.5, 0.8, -0.5, 0],
    lowrank_init=[0.8, 0.5, 0.8, 0]
)

session.run(tf.global_variables_initializer())
K_gt_tensor = session.run(K_gt.to_dense())
matshow(K_gt_tensor)

C_gt = tf.linalg.inv(K_gt_tensor)
# matshow(session.run(C_gt)

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance


model_gt = MultivariateNormalFullCovariance(loc=tf.zeros([C_gt.shape[0]], dtype=np.float32),
                                            covariance_matrix=C_gt)
data_train = session.run(model_gt.sample(50))
data_train = tf.constant(data_train)

matshow(session.run(tf.linalg.inv(tf.matmul(tf.transpose(data_train), data_train))));

data_test = session.run(model_gt.sample(10000))
data_test = tf.constant(data_test)
matshow(session.run(tf.linalg.inv(tf.matmul(tf.transpose(data_test), data_test))));

K, variables = make_block_linear_operator(block_sizes)
session.run(tf.global_variables_initializer())
matshow(session.run(K.to_dense()))

def loss(K, data):
    return tf.reduce_mean(0.5 * (
        - K.log_abs_determinant()
        + tf.reduce_sum(tf.transpose(data) * K.matmul(tf.transpose(data)), axis=0)
        + K.shape[0].value * np.log(2 * np.pi)
    ))

session.run(loss(K_gt, data_train))

session.run(loss(K_gt, data_test))

train_loss = loss(K, data_train)
session.run(train_loss)

test_loss = loss(K, data_test)
session.run(test_loss)

session.run(variables)

grads_op = tf.gradients(loss(K, data_train), variables)
grads_value = session.run(grads_op)
grads_value

diag_grad_norm, low_rank_grad_norm = [np.linalg.norm(gv) for gv in grads_value]

diag_grad_norm

low_rank_grad_norm

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(train_loss, var_list=variables)
session.run(tf.global_variables_initializer())

history = {
    'train_loss': [session.run(train_loss)],
    'test_loss': [session.run(test_loss)],
    'diag_grad_norm': [diag_grad_norm],
    'low_rank_grad_norm': [low_rank_grad_norm],
    'step': [0],
}
for i in range(250):
    _, train_loss_value, grads_value = session.run([train_op, train_loss, grads_op])
    if (i + 1) % 10 == 0:
        test_loss_value = session.run(test_loss)
        diag_grad_norm, low_rank_grad_norm = [np.linalg.norm(gv) for gv in grads_value]
        print(f"[{i + 1:03d}] train loss: {train_loss_value:.5f}, test loss: {test_loss_value:.5f}, "
              f"diag grad: {diag_grad_norm:.5f}, lowrank grad: {low_rank_grad_norm:.5f}")
        history['train_loss'].append(train_loss_value)
        history['test_loss'].append(test_loss_value)
        history['diag_grad_norm'].append(diag_grad_norm)
        history['low_rank_grad_norm'].append(low_rank_grad_norm)
        history['step'].append(i + 1)
        
        # TODO: snapshot parameters for model with best validation set params
        # Early stop when no more progress on training set (just to
        # keep convergence plots interesting and highlight overfitting).
        
        # TODO: Polyak-Rupert averaging of the estimator variables + add small noise
        # to training data to make Polyak-Rupert averaging more interesting.

fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 8))
x = history['step']
ax0.plot(x, history['train_loss'], label='train')
ax0.plot(x, history['test_loss'], label='test')
ax0.set_ylabel('negative log likelihood')
ax0.set_xticks([])
ax0.legend()

ax1.plot(x, history['diag_grad_norm'], label='diagonal parameters')
ax1.plot(x, history['low_rank_grad_norm'], label='lowrank parameters')
ax1.set_ylabel('gradient norms')
ax1.set_xlabel('steps')
ax1.legend();

history['diag_grad_norm'][-1]

history['low_rank_grad_norm'][-1]

matshow(session.run(K.to_dense()))

matshow(session.run(K_gt.to_dense()))

# try:
#     tfe.enable_eager_execution()
# except ValueError:
#     # Hide the annoying lack of idem-potency.
#     pass

block_boundaries = [0, 5, 8, 10]
# block_boundaries = [0]
# block_boundaries.append(block_boundaries[-1] + 784 * 10)
# block_boundaries.append(block_boundaries[-1] + 10)
# block_boundaries.append(block_boundaries[-1] + 10 * 10)
# block_boundaries.append(block_boundaries[-1] + 10)
# block_boundaries.append(block_boundaries[-1] + 10 * 10)
# block_boundaries.append(block_boundaries[-1] + 10)



def outer(v):
    return tf.matmul(tf.reshape(v, (-1, 1)),
                     tf.reshape(v, (1, -1)))


def parametrized_precision(block_boundaries):
    variables = []
    n_params = block_boundaries[-1]
    K = tf.zeros(shape=(n_params, n_params))
    for i, j in zip(block_boundaries, block_boundaries[1:]):
        # Diagonal for the current block
        diag_mask_i = np.zeros(n_params, dtype=np.float32)
        diag_mask_i[i:j] = 1
        diag_mask_i = tf.constant(diag_mask_i)
        K_i_var = tf.Variable(0.5, dtype=np.float32)
        K_i = tf.diag(tf.exp(K_i_var) * diag_mask_i)
        variables.append(K_i_var)
        K += K_i
        
        # Rank-one for the current parameter block
        ro_mask_i = np.zeros(n_params, dtype=np.float32)
        ro_mask_i[i:j] = 1
        ro_mask_i = tf.constant(ro_mask_i)
        ro_var_i = tf.Variable(1, dtype=np.float32)
        ro_i = outer(ro_var_i * ro_mask_i)
        variables.append(ro_var_i)
        K += ro_i
    for i, j in zip(block_boundaries, block_boundaries[2:]):
        # Rank-one for consecutive blocks interactions
        ro_mask_i = np.zeros(n_params, dtype=np.float32)
        ro_mask_i[i:j] = 1
        ro_mask_i = tf.constant(ro_mask_i)
        ro_var_i = tf.Variable(1., dtype=np.float32)
        ro_i = outer(ro_var_i * ro_mask_i)
        variables.append(ro_var_i)
        K += ro_i
    return K, variables

K2, variables = parametrized_precision(block_boundaries)

session.run(tf.global_variables_initializer())
K2 = session.run(K2)

matshow(K2)
len(variables)

np.linalg.det(K2)

eigvals = session.run(tf.linalg.eigh(K2)[0])
plt.bar(np.arange(K2.shape[0]), eigvals)
eigvals

C = session.run(tf.linalg.inv(K2))
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.matshow(C)

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance


model = MultivariateNormalFullCovariance(loc=tf.zeros([C.shape[0]], dtype=np.float32), covariance_matrix=C)

data = session.run(model.sample(100))
matshow(session.run(tf.linalg.inv(tf.matmul(tf.transpose(data), data))));

# from sklearn.covariance import graph_lasso

# gl = graph_lasso(np.asarray(tf.matmul(tf.transpose(data), data)), 1)
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.matshow(gl[1])

data_large = session.run(model.sample(10000))
matshow(session.run(tf.linalg.inv(tf.matmul(tf.transpose(data_large), data_large))));

from numpy.linalg import qr


def unitary(rng, n, m, dtype=np.float32):
    if n <= m:
        return qr(rng.randn(m, n).astype(dtype), mode='reduced')[0].T
    else:
        return qr(rng.randn(n, m).astype(dtype), mode='reduced')[0]


def ill_cond(rng, n, m, cond_number=100, dtype=np.float32):
    if n <= m:
        u = unitary(rng, n, n, dtype=dtype)
        d = np.diag(np.linspace(1, cond_number, n, dtype=dtype))
        v = unitary(rng, n, m, dtype=dtype)
    else:
        u = unitary(rng, n, m, dtype=dtype)
        d = np.diag(np.linspace(1, cond_number, m, dtype=dtype))
        v = unitary(rng, m, m, dtype=dtype)
    return u @ d @ v

rng = np.random.RandomState(42)
u = unitary(rng, 3, 5)
u.shape, np.linalg.cond(u)

v = unitary(rng, 5, 3)
v.shape, np.linalg.cond(v)

a = ill_cond(rng, 3, 5)
a.shape, np.linalg.cond(a)

b = ill_cond(rng, 5, 3)
b.shape, np.linalg.cond(b)

n_features = 2
hidden_dim = 8
n_outputs = 3
rng = np.random.RandomState(42)
W1 = unitary(rng, n_features, hidden_dim)
W2 = ill_cond(rng, hidden_dim, hidden_dim, cond_number=10)
# W3 = unitary(rng, hidden_dim, hidden_dim)
W4 = ill_cond(rng, hidden_dim, n_outputs, cond_number=1)
W = W1 @ W2 @ W4
np.linalg.cond(W)

n_samples_train = 100000
n_samples_test = 60000

X_train = rng.randn(n_samples_train, n_features)
X_test = rng.randn(n_samples_test, n_features)

y_train = (X_train @ W).argmax(axis=-1)
y_test = (X_test @ W).argmax(axis=-1)

class DeepLinearModel:

    def __init__(self, params=[], dtype=tf.float32,):
        self.variables = [tf.Variable(v, dtype=dtype) for v in params]
        self._input_features = tf.placeholder(
            dtype, shape=(None, n_features), name='input_features')
        self._target_labels = tf.placeholder(
            tf.int32, name='target_labels')

        x = self._input_features
        for v in self.variables:
            x = tf.matmul(x, v)
        self._logits = x
        self._probabilities = tf.nn.softmax(self._logits)
        self._loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self._target_labels, logits=self._logits)
        

dlm_gt = DeepLinearModel(params=[W1, W2, W4])
session.run(tf.variables_initializer(dlm_gt.variables))

probs_gt_train = session.run(dlm_gt._probabilities,
                             feed_dict={dlm_gt._input_features: X_train})
y_sampled_by_gt = np.asarray([rng.choice(np.arange(n_outputs, dtype=np.int32), p=p)
                              for p in probs_gt_train])

batch_size = 32
grads_op = tf.gradients(dlm_gt._loss, dlm_gt.variables)

collected_gradients = []

for idx in range(0, X_train.shape[0], batch_size):
    grads = session.run(grads_op, feed_dict={
        dlm_gt._input_features: X_train[idx:idx + batch_size],
        dlm_gt._target_labels: y_sampled_by_gt[idx:idx + batch_size],
    })
    collected_gradients.append(np.concatenate([g.ravel() for g in grads]))
    
collected_gradients = np.asarray(collected_gradients)
collected_gradients.shape

prec_mat = np.linalg.inv(collected_gradients.T @ collected_gradients)
matshow(prec_mat)

matshow(prec_mat[:2 * 8, :2 * 8])

last_blocks = 3 * 8 + 3
matshow(prec_mat[-last_blocks:, -last_blocks:])



