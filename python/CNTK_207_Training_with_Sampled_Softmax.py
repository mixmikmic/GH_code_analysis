import os
import cntk as C

# Select the right target device when this notebook is being tested:
C.device.try_set_default_device(C.device.gpu(0))

from __future__ import print_function
from __future__ import division


# Creates a subgraph computing cross-entropy with sampled softmax.
def cross_entropy_with_sampled_softmax_and_embedding(
    hidden_vector,            # Node providing hidden input
    target_vector,            # Node providing the expected labels (as sparse vectors)
    num_classes,              # Number of classes
    hidden_dim,               # Dimension of the hidden vector
    num_samples,              # Number of samples to use for sampled softmax
    sampling_weights,         # Node providing weights to be used for the weighted sampling
    allow_duplicates = True,  # Boolean flag to control whether to use sampling with replacemement 
                              # (allow_duplicates == True) or without replacement.
    ):
    # define the parameters learnable parameters
    b = C.Parameter(shape = (num_classes, 1), init = 0)
    W = C.Parameter(shape = (num_classes, hidden_dim), init = C.glorot_uniform())

    # Define the node that generates a set of random samples per minibatch
    # Sparse matrix (num_samples * num_classes)
    sample_selector = C.random_sample(sampling_weights, num_samples, allow_duplicates)

    # For each of the samples we also need the probablity that it in the sampled set.
    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, num_samples, allow_duplicates) # dense row [1 * vocab_size]
    log_prior = C.log(inclusion_probs) # dense row [1 * num_classes]

    # Create a submatrix wS of 'weights
    W_sampled = C.times(sample_selector, W) # [num_samples * hidden_dim]
    z_sampled = C.times_transpose(W_sampled, hidden_vector) + C.times(sample_selector, b) - C.times_transpose (sample_selector, log_prior)# [num_samples]

    # Getting the weight vector for the true label. Dimension hidden_dim
    W_target = C.times(target_vector, W) # [1 * hidden_dim]
    z_target = C.times_transpose(W_target, hidden_vector) + C.times(target_vector, b) - C.times_transpose(target_vector, log_prior) # [1]


    z_reduced = C.reduce_log_sum_exp(z_sampled)
    
    # Compute the cross entropy that is used for training.
    # We don't check whether any of the classes in the random samples conincides with the true label, so it might
    # happen that the true class is counted
    # twice in the normalising demnominator of sampled softmax.
    cross_entropy_on_samples = C.log_add_exp(z_target, z_reduced) - z_target

    # For applying the model we also output a node providing the input for the full softmax
    z = C.times_transpose(W, hidden_vector) + b
    z = C.reshape(z, shape = (num_classes))

    zSMax = C.reduce_max(z_sampled)
    error_on_samples = C.less(z_target, zSMax)
    return (z, cross_entropy_on_samples, error_on_samples)



# Creates subgraph computing cross-entropy with (full) softmax.
def cross_entropy_with_softmax_and_embedding(
    hidden_vector,  # Node providing hidden input
    target_vector,  # Node providing the expected labels (as sparse vectors)
    num_classes,    # Number of classes
    hidden_dim      # Dimension of the hidden vector
    ):
    # Setup bias and weights
    b = C.Parameter(shape = (num_classes, 1), init = 0)
    W = C.Parameter(shape = (num_classes, hidden_dim), init = C.glorot_uniform())

    
    z = C.reshape( C.times_transpose(W, hidden_vector) + b, (1, num_classes))
    
    # Use cross_entropy_with_softmax
    cross_entropy = C.cross_entropy_with_softmax(z, target_vector)

    zMax = C.reduce_max(z)
    zT = C.times_transpose(z, target_vector)
    error_on_samples = C.less(zT, zMax)

    return (z, cross_entropy, error_on_samples)

import numpy as np
from math import log, exp, sqrt
from cntk.logging import ProgressPrinter
import timeit


# A class with all parameters
class Param:
    # Learning parameters
    learning_rate = 0.03
    minibatch_size = 100
    num_minbatches = 100
    test_set_size = 1000
    momentum_time_constant = 5 * minibatch_size
    reporting_interval = 10
    allow_duplicates = False
    
    # Parameters for sampled softmax
    use_sampled_softmax = True
    use_sparse = True
    softmax_sample_size = 10

    # Details of data and model
    num_classes = 50
    hidden_dim = 10
    
data_sampling_distribution = lambda: np.repeat(1.0 / Param.num_classes, Param.num_classes)
    
softmax_sampling_weights = lambda: np.repeat(1.0 / Param.num_classes, Param.num_classes)


# Creates random one-hot vectors of dimension 'num_classes'.
# Returns a tuple with a list of one-hot vectors, and list with the indices they encode.
def get_random_one_hot_data(num_vectors):
    indices = np.random.choice(
        range(Param.num_classes),
        size=num_vectors, 
        p = data_sampling_distribution()).reshape((num_vectors, 1))
    list_of_vectors = C.Value.one_hot(indices, Param.num_classes)
    return (list_of_vectors, indices.flatten())

# Create a network that:
# * Transforms the input one hot-vectors with a constant random embedding
# * Applies a linear decoding with parameters we want to learn
def create_model(labels):
    # random projection matrix
    random_data = np.random.normal(scale = sqrt(1.0/Param.hidden_dim), size=(Param.num_classes, Param.hidden_dim)).astype(np.float32)
    random_matrix = C.constant(shape = (Param.num_classes, Param.hidden_dim), value = random_data)
    
    h = C.times(labels, random_matrix)
    
    # Connect the latent output to (sampled/full) softmax.
    if Param.use_sampled_softmax:
        sampling_weights = np.asarray(softmax_sampling_weights(), dtype=np.float32)
        sampling_weights.reshape((1, Param.num_classes))
        softmax_input, ce, errs = cross_entropy_with_sampled_softmax_and_embedding(
            h, 
            labels,
            Param.num_classes, 
            Param.hidden_dim, 
            Param.softmax_sample_size, 
            softmax_sampling_weights(),
            Param.allow_duplicates)
    else:
        softmax_input, ce, errs = cross_entropy_with_softmax_and_embedding(
            h, 
            labels, 
            Param.num_classes, 
            Param.hidden_dim)

    return softmax_input, ce, errs

def train(do_print_progress):
    labels = C.input(shape = Param.num_classes, is_sparse = Param.use_sparse)
    z, cross_entropy, errs = create_model(labels)

    # Setup the trainer
    learning_rate_schedule = C.learning_rate_schedule(Param.learning_rate, C.UnitType.sample)
    momentum_schedule = C.momentum_as_time_constant_schedule(Param.momentum_time_constant)
    learner = C.momentum_sgd(z.parameters, learning_rate_schedule, momentum_schedule, True)
    progress_writers = None
    if do_print_progress:
        progress_writers = [ProgressPrinter(freq=Param.reporting_interval, tag='Training')]
    trainer = C.Trainer(z, (cross_entropy, errs), learner, progress_writers)

    minbatch = 0
    average_cross_entropy = compute_average_cross_entropy(z)
    minbatch_data = [0] # store minibatch values
    cross_entropy_data = [average_cross_entropy] # store cross_entropy values

    # Run training
    t_total= 0

    # Run training
    for minbatch in range(1,Param.num_minbatches):
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        label_data, indices = get_random_one_hot_data(Param.minibatch_size)
        arguments = ({labels : label_data})

        # If do_print_progress is True, this will automatically print the progress using ProgressPrinter
        # The printed loss numbers are computed using the sampled softmax criterion
        t_start = timeit.default_timer()
        trainer.train_minibatch(arguments)
        t_end = timeit.default_timer()

        t_delta = t_end - t_start
        samples_per_second = Param.minibatch_size / t_delta
        
        # We ignore the time measurements of the first two minibatches
        if minbatch > 2:
            t_total += t_delta

        # For comparison also print result using the full criterion
        if minbatch % Param.reporting_interval == int(Param.reporting_interval/2):
            # memorize the progress data for plotting
            average_cross_entropy = compute_average_cross_entropy(z)
            minbatch_data.append(minbatch)
            cross_entropy_data.append(average_cross_entropy)
            
            if do_print_progress:
                print("\nMinbatch=%d Cross-entropy from full softmax = %.3f perplexity = %.3f samples/s = %.1f"
                    % (minbatch, average_cross_entropy, exp(average_cross_entropy), samples_per_second))
                
    # Number of samples we measured. First two minbatches were ignored
    samples_measured = Param.minibatch_size * (Param.num_minbatches - 2)
    overall_samples_per_second = samples_measured / t_total
    return (minbatch_data, cross_entropy_data, overall_samples_per_second) 

def compute_average_cross_entropy(softmax_input):
    vectors, indices = get_random_one_hot_data(Param.test_set_size)
    total_cross_entropy = 0.0
    arguments = (vectors)
    z = softmax_input.eval(arguments).reshape(Param.test_set_size, Param.num_classes)

    for i in range(len(indices)):
        log_p = log_softmax(z[i], indices[i])
        total_cross_entropy -= log_p

    return total_cross_entropy / len(indices)

# Computes log(softmax(z,index)) for a one-dimensional numpy array z in an numerically stable way.
def log_softmax(z,    # numpy array
                index # index into the array
            ):
    max_z = np.max(z)
    return z[index] - max_z - log(np.sum(np.exp(z - max_z)))



np.random.seed(1)

print("start...")
train(do_print_progress = True)
print("done.")

# We want to lot the data 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Define weights of zipfian distributuion
def zipf(index):
    return 1.0 / (index + 5)

# Use zipifian distribution for the classes
def zipf_sampling_weights():
    return np.asarray([ zipf(i) for i in range(Param.num_classes)], dtype=np.float32)

data_sampling_distribution = lambda: zipf_sampling_weights() / np.sum(zipf_sampling_weights())

print("start...")


# Train using uniform sampling (like before)
np.random.seed(1)
softmax_sampling_weights = lambda: np.repeat(1.0/Param.num_classes, Param.num_classes)
minibatch_data, cross_entropy_data, _ = train(do_print_progress = False)

# Train using importance sampling
np.random.seed(1)
softmax_sampling_weights = zipf_sampling_weights
minibatch_data2, cross_entropy_data2, _ = train(do_print_progress = False)

plt.plot(minibatch_data, cross_entropy_data, 'r--',minibatch_data, cross_entropy_data2, 'b--')
plt.xlabel('number of mini-batches')
plt.ylabel('cross entropy')
plt.show()


print("start...")

# Reset parameters
class Param:
    # Learning parameters
    learning_rate = 0.03
    minibatch_size = 8
    num_minbatches = 100
    test_set_size = 1 # we are only interrested in speed
    momentum_time_constant = 5 * minibatch_size
    reporting_interval = 1000000 # Switch off reporting to speed up
    allow_duplicates = False
    
    # Parameters for sampled softmax
    use_sampled_softmax = True
    use_sparse = True
    softmax_sample_size = 10

    # Details of data and model
    num_classes = 50000
    hidden_dim = 10
    
data_sampling_distribution = lambda: np.repeat(1.0 / Param.num_classes, Param.num_classes)
softmax_sampling_weights = lambda: np.repeat(1.0 / Param.num_classes, Param.num_classes)

    
sample_sizes = [5, 10, 100, 1000]
speed_with_sampled_softmax = []

# Get the speed with sampled softmax for different sizes
for sample_size in sample_sizes: 
    print("Measuring speed of sampled softmax for sample size %d ..." % (sample_size))
    Param.use_sampled_softmax = True
    Param.softmax_sample_size = sample_size
    _, _,  samples_per_second = train(do_print_progress = False)
    speed_with_sampled_softmax.append(samples_per_second)

# Get the speed with full softmax
Param.use_sampled_softmax = False
print("Measuring speed of full softmax ...")
_, _,  samples_per_second = train(do_print_progress = False)
speed_without_sampled_softmax = np.repeat(samples_per_second, len(sample_sizes))

# Plot the speed of sampled softmax (blue) as a function of sample sizes
# and compare it to the speed with full softmax (red).    
plt.plot(sample_sizes, speed_without_sampled_softmax, 'r--',sample_sizes, speed_with_sampled_softmax, 'b--')
plt.xlabel('softmax sample size')
plt.ylabel('speed: instances / second')
plt.title("Speed 'sampled softmax' (blue) vs. 'full softmax' (red)")
plt.ylim(ymin=0)
plt.show()

