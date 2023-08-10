get_ipython().run_line_magic('matplotlib', 'notebook')
import pylab
pylab.rcParams['figure.figsize'] = (8, 4)
from pylab import *

def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

from scipy.optimize import curve_fit

mean_hunger = 5
samples_per_day = 100
n_days = 10000
samples = np.random.normal(loc=mean_hunger, size=(n_days, samples_per_day))
daily_maxes = np.max(samples, axis=1)

def gumbel_pdf(prob, loc, scale):
    z = (prob - loc) / scale
    return exp(-z - exp(-z)) / scale

def plot_maxes(daily_maxes):
    probs, hungers, _ = hist(daily_maxes, normed=True, bins=100)
    xlabel("Hunger")
    ylabel("Probability of hunger being daily maximum")
    
    (loc, scale), _ = curve_fit(gumbel_pdf, hungers[:-1], probs)
    plot(hungers, gumbel_pdf(hungers, loc, scale))

figure()
plot_maxes(daily_maxes)

most_likely_hunger = 5
samples = np.random.exponential(scale=most_likely_hunger,
                                size=(n_days, samples_per_day))
daily_maxes = np.max(samples, axis=1)

figure()
plot_maxes(daily_maxes)

n_cats = 7
cats = np.arange(n_cats)

probs = np.random.randint(low=1, high=20, size=n_cats)
probs = probs / sum(probs)
logits = log(probs)

def plot_probs():
    bar(cats, probs)
    xlabel("Category")
    ylabel("Probability")
    
figure()
plot_probs()

n_samples = 1000

def plot_estimated_probs(samples):
    n_cats = np.max(samples) + 1
    estd_probs, _, _ = hist(samples,
                            bins=np.arange(n_cats + 1),
                            align='left',
                            edgecolor='white',
                            normed=True)
    xlabel("Category")
    ylabel("Estimated probability")
    return estd_probs

def print_probs(probs):
    print(" ".join(["{:.2f}"] * len(probs)).format(*probs))

samples = np.random.choice(cats, p=probs, size=n_samples)

figure()
subplot(1, 2, 1)
plot_probs()
subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

def sample(logits):
    noise = np.random.uniform(size=len(logits))
    sample = np.argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

figure()
subplot(1, 2, 1)
plot_probs()
subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

def sample(logits):
    noise = np.random.normal(size=len(logits))
    sample = argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

figure()
subplot(1, 2, 1)
plot_probs()
subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

def sample(logits):
    noise = np.random.gumbel(size=len(logits))
    sample = argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

figure()
subplot(1, 2, 1)
plot_probs()
subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

numpy_gumbel = np.random.gumbel(size=n_samples)
manual_gumbel = -log(-log(np.random.uniform(size=n_samples)))
figure()
subplot(1, 2, 1)
hist(numpy_gumbel, bins=50, normed=True)
ylabel("Probability")
xlabel("numpy Gumbel")
subplot(1, 2, 2)
hist(manual_gumbel, bins=50, normed=True)
xlabel("Gumbel from uniform noise");

import tensorflow as tf
sess = tf.Session()

def differentiable_sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    logits_with_noise = logits - tf.log(-tf.log(noise))
    return tf.nn.softmax(logits_with_noise)

mean = tf.Variable(2.)
idxs = tf.Variable([0., 1., 2., 3., 4.])
# An unnormalised approximately-normal distribution
logits = tf.exp(-(idxs - mean) ** 2)
sess.run(tf.global_variables_initializer())

def print_logit_vals():
    logit_vals = sess.run(logits)
    print(" ".join(["{:.2f}"] * len(logit_vals)).format(*logit_vals))
    
print("Logits: ", end="")
print_logit_vals()

sample = differentiable_sample(logits)
sample_weights = tf.Variable([1., 2., 3., 4., 5.], trainable=False)
result = tf.reduce_sum(sample * sample_weights)

sess.run(tf.global_variables_initializer())
train_op = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(-result)

print("Distribution mean: {:.2f}".format(sess.run(mean)))
for i in range(5):
    sess.run(train_op)
    print("Distribution mean: {:.2f}".format(sess.run(mean)))

