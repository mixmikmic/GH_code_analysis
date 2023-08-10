get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pymc3 as pm

import theano.tensor as tt
import theano

pm.__version__

data = 10.
# prior for mu and sd
prior = [0., 100.]

theano.config.compute_test_value='off'
mu_th = tt.scalar('mu')
logp_th = pm.Normal.dist(mu=prior[0], sd=prior[1]).logp(mu_th)
logp_th += pm.Normal.dist(mu_th, 1.).logp(data)

logp_th.eval({mu_th: 0.})

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.__version__

mu0 = tf.Variable(0., tf.float64)

mu_tf = tfd.Normal(loc=prior[0], scale=prior[1])

mu_tf.log_prob(mu0)

y_tf = tfd.Normal(loc=mu0, scale=1.)
logp_tf = y_tf.log_prob(data) + mu_tf.log_prob(mu0)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(logp_tf))

scale = 1.
nsample = 10000

get_ipython().run_cell_magic('time', '', "trace = dict(mu=np.zeros(nsample),\n             logp=np.zeros(nsample),\n             accept=np.zeros(nsample))\n\n# logp0 = sess.run(logp_tf, feed_dict={mu0: 0.})\nwith sess.as_default():\n    logp0 = logp_tf.eval({mu0: 0.})\n\ntrace['mu'][0] = 0.\ntrace['logp'][0] = logp0\ntrace['accept'][0] = 1\n\nfor s in range(1, nsample):\n    mu_ = trace['mu'][s-1]\n    logp_ = trace['logp'][s-1]\n    mu_p = mu_ + np.random.randn()*scale\n\n    # evaluation of logp\n    logp_p = sess.run(logp_tf, feed_dict={mu0: mu_p})\n#     with sess.as_default():\n#         logp_p = logp_tf.eval({mu0: mu_p})\n\n    if np.log(np.random.rand()) < logp_p-logp_:\n        trace['mu'][s] = mu_p\n        trace['logp'][s] = logp_p\n        trace['accept'][s] = 1\n    else:\n        trace['mu'][s] = mu_\n        trace['logp'][s] = logp_\n        trace['accept'][s] = 0")

trace_tf = trace
_, ax = plt.subplots(1, 3, figsize=(15, 2))
pm.kdeplot(trace['mu'], ax=ax[0])
ax[1].plot(trace['mu'])
ax[1].set_title('Mean acceptance ratio: {:.5f}'
                .format(trace['accept'].mean()))
ax[2].plot(trace['logp'])
plt.tight_layout();

get_ipython().run_cell_magic('time', '', "trace = dict(mu=np.zeros(nsample),\n             logp=np.zeros(nsample),\n             accept=np.zeros(nsample))\n\nlogp0 = logp_th.eval({mu_th: 0.})\n  \ntrace['mu'][0] = 0.\ntrace['logp'][0] = logp0\ntrace['accept'][0] = 1\n\nfor s in range(1, nsample):\n    mu_ = trace['mu'][s-1]\n    logp_ = trace['logp'][s-1]\n    mu_p = mu_ + np.random.randn()*scale\n\n    # evaluation of logp\n    logp_p = logp_th.eval({mu_th: mu_p})\n\n    if np.log(np.random.rand()) < logp_p-logp_:\n        trace['mu'][s] = mu_p\n        trace['logp'][s] = logp_p\n        trace['accept'][s] = 1\n    else:\n        trace['mu'][s] = mu_\n        trace['logp'][s] = logp_\n        trace['accept'][s] = 0")

trace_th = trace
_, ax = plt.subplots(1, 3, figsize=(15, 2))
pm.kdeplot(trace['mu'], ax=ax[0])
ax[1].plot(trace['mu'])
ax[1].set_title('Mean acceptance ratio: {:.5f}'
                .format(trace['accept'].mean()))
ax[2].plot(trace['logp'])
plt.tight_layout();

from tensorflow_probability.python.mcmc.random_walk_metropolis import random_walk_normal_fn
from tensorflow_probability import edward2 as ed

dtype = np.float32
def target_log_prob(x):
    latent = tfd.Normal(loc=dtype(0), scale=dtype(100))
    y = tfd.Normal(loc=x, scale=dtype(1))
    return latent.log_prob(x)+y.log_prob(data)

samples, _ = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=dtype(1),
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target_log_prob,
        seed=42),
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.
sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
with tf.Session() as sess:
    [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

print('Estimated mean: {}'.format(sample_mean_))
print('Estimated standard deviation: {}'.format(sample_std_))

tf.reset_default_graph()

def Normal_model():
    x = ed.Normal(loc=0., scale=100., name='x')
    y = ed.Normal(loc=x, scale=1., name='y')
    return y

model_template = tf.make_template('Normal_model', Normal_model)
log_joint = ed.make_log_joint_fn(model_template)

def target_log_prob_fn(x):
    """Unnormalized target density as a function of states."""
    return log_joint(x=x, y=data)

get_ipython().run_cell_magic('time', '', "trace = dict(mu=np.zeros(nsample),\n             logp=np.zeros(nsample),\n             accept=np.zeros(nsample))\n\nsamples, kernelresult = tfp.mcmc.sample_chain(\n    num_results=nsample,\n    current_state=tf.zeros([], name='x'),\n    kernel=tfp.mcmc.RandomWalkMetropolis(\n        target_log_prob_fn,\n        seed=42),\n    num_burnin_steps=0,\n    parallel_iterations=1)  # For determinism.")

with tf.Session() as sess:
    [
        trace['mu'],
        trace['logp'],
        trace['accept']
    ] = sess.run(
        [samples,
         target_log_prob(samples),
         kernelresult.is_accepted
         ])

trace_tf1 = trace
_, ax = plt.subplots(1, 3, figsize=(15, 2))
pm.kdeplot(trace['mu'], ax=ax[0])
ax[1].plot(trace['mu'])
ax[2].plot(trace['logp'])
plt.tight_layout();

kernel=tfp.mcmc.RandomWalkMetropolis(
       target_log_prob,
       seed=42)

state = tf.zeros([], name='x')
previous_kernel_results = kernel.bootstrap_results(state)

kernel.one_step(state, previous_kernel_results)

get_ipython().run_cell_magic('time', '', 'ndim = 100\nnstep = 10000\n\ntf.reset_default_graph()\ni0 = tf.constant(0)\nm0 = tf.ones([1, ndim])\nwalk = tf.random_normal([1, ndim], mean=0., stddev=1.)\nc = lambda i, m: i < nstep\nb = lambda i, m: [i+1, \n                  tf.concat([m, m[-1]+tf.random_normal([1, ndim], mean=0., stddev=1.)], \n                            axis=0)]\nr = tf.while_loop(\n    c, b, loop_vars=[i0, m0],\n    shape_invariants=[i0.get_shape(), tf.TensorShape([None, ndim])])\n\ninit = tf.global_variables_initializer()\nwith tf.Session() as sess:\n    sess.run(init)\n    r_ = sess.run(r)')

plt.plot(r_[1], alpha=.1);

tf.reset_default_graph()
# Generate nsample initial values randomly. Each of these would be an
# independent starting point for a Markov chain.
state = variable_scope.get_variable('state', initializer=[0.0])

# Computes the log(p(x))
def log_density(x):
    mu = tfd.Normal(loc=0., scale=100.)
    y = tfd.Normal(loc=x, scale=1.)
    logp2 = y.log_prob(data)+mu.log_prob(x)
    return logp2

# Initial log-density value
state_log_density = tf.get_variable(
    "state_log_density",
    initializer=log_density(state.initialized_value()))

# A variable to store the log_acceptance_ratio:
log_acceptance_ratio = tf.get_variable(
    "log_acceptance_ratio",
    initializer=tf.zeros([1], dtype=tf.float32))

# Generates random proposals
def random_proposal(x):
      return (x + tf.random_normal(tf.shape(x), mean=0., stddev=scale,
                                dtype=x.dtype, seed=12)), None

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

def mhevolve(initial_sample,
           initial_log_density,
           initial_log_accept_ratio,
           target_log_prob_fn,
           proposal_fn,
           n_steps=1,
           seed=None,
           name=None):
    with ops.name_scope(name, "metropolis_hastings", [initial_sample]):
        current_state = tf.expand_dims(initial_sample, 0)
        current_target_log_prob = tf.expand_dims(initial_log_density, 0)
        log_accept_ratio = initial_log_accept_ratio

        def step(i, current_state, current_target_log_prob, log_accept_ratio):
            """Wrap single Markov chain iteration in `while_loop`."""
            next_state, kernel_results = mh.kernel(
                      target_log_prob_fn=target_log_prob_fn,
                      proposal_fn=proposal_fn,
                      current_state=current_state[-1, :],
                      current_target_log_prob=current_target_log_prob[-1, :],
                      seed=seed)
            accepted_log_prob = kernel_results.current_target_log_prob
            log_accept_ratio = kernel_results.log_accept_ratio
            current_state = tf.concat([current_state, tf.expand_dims(next_state, 0)], 
                            axis=0)
            current_target_log_prob = tf.concat([current_target_log_prob,
                                                 tf.expand_dims(accepted_log_prob, 0)], 
                            axis=0)
            return i + 1, current_state, current_target_log_prob, log_accept_ratio
        
        i0 = tf.constant(0)
        
        (_, accepted_state, accepted_target_log_prob, accepted_log_accept_ratio) = (
            control_flow_ops.while_loop(
                cond=lambda i, *ignored_args: i < n_steps,
                body=step,
                loop_vars=[
                    i0,  # i
                    current_state,
                    current_target_log_prob,
                    log_accept_ratio,
                ],
                # the magic here
                shape_invariants=[
                    i0.get_shape(),
                    tf.TensorShape([None, 1]),
                    tf.TensorShape([None, 1]),
                    log_accept_ratio.get_shape(),
                ],
                parallel_iterations=1 if seed is not None else 10,
                # TODO(b/73775595): Confirm optimal setting of swap_memory.
                swap_memory=1))

        forward_step = control_flow_ops.group(
            accepted_target_log_prob,
            accepted_state,
            accepted_log_accept_ratio
        )

    return accepted_state, accepted_target_log_prob, accepted_log_accept_ratio

get_ipython().run_cell_magic('time', '', '#  Create the op to propagate the chain for 100 steps.\naccepted_state, accepted_target_log_prob, accepted_log_accept_ratio = mhevolve(\n    state, state_log_density, log_acceptance_ratio,\n    log_density, random_proposal, n_steps=nsample, seed=123)\n\ninit = tf.global_variables_initializer()\nwith tf.Session() as sess:\n    sess.run(init)\n    # Run the chains for a total of 1000 steps\n    # Executing the stepper advances the chain to the next state.\n    samples, alog_prob, a_ratio = sess.run(\n        [accepted_state, accepted_target_log_prob, accepted_log_accept_ratio])')

trace = dict(mu=np.zeros(nsample),
             logp=np.zeros(nsample),
             accept=np.zeros(nsample))

trace['mu'] = samples[1:].squeeze()
trace['logp'] = alog_prob[1:].squeeze()
trace['accept'][1:] = trace['mu'][1:] != trace['mu'][:-1]

trace_tf3 = trace
_, ax = plt.subplots(1, 3, figsize=(15, 2))
pm.kdeplot(trace['mu'], ax=ax[0])
ax[1].plot(trace['mu'])
ax[1].set_title('Mean acceptance ratio: {:.5f}'
                .format(trace['accept'].mean()))
ax[2].plot(trace['logp'])
plt.tight_layout();

