import helpers
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

population_data = helpers.distribution(50, 10000, 100)
helpers.histogram_visualization(population_data)
print('population mean ', np.mean(population_data))

def random_sample(population_data, sample_size):
    return np.random.choice(population_data, size = sample_size)

random_sample(population_data, 10)

def sample_mean(sample):
    return np.mean(sample)

# take a sample from the population
example_sample = random_sample(population_data, 10)

# calculate the mean of the sample and output the results
sample_mean(example_sample)

###
# Code for showing how the central limit theorem works.
# The function inputs:
# population - population data
# n - sample size
# iterations - number of times to draw random samples

def central_limit_theorem(population, n, iterations):
    sample_means_results = []
    for i in range(iterations):
        # get a random sample from the population of size n
        sample = random_sample(population, n)
        
        # calculate the mean of the random sample 
        # and append the mean to the results list
        sample_means_results.append(sample_mean(sample))
    return sample_means_results

print('Means of all the samples ')
central_limit_theorem(population_data, 10, 10000)

import matplotlib.pyplot as plt

def visualize_results(sample_means):

    plt.hist(sample_means, bins = 30)
    plt.title('Histogram of the Sample Means')
    plt.xlabel('Mean Value')
    plt.ylabel('Count')

# Take random sample and calculate the means
sample_means_results = central_limit_theorem(population_data, 30, 10000)

# Visualize the results
visualize_results(sample_means_results)

# Take random sample and calculate the means
sample_means_results = central_limit_theorem(population_data, 1, 10000)

# Visualize the results
visualize_results(sample_means_results)

# Take random sample and calculate the means
sample_means_results = central_limit_theorem(population_data, 10, 10000)

# Visualize the results
visualize_results(sample_means_results)

# Take random sample and calculate the means
sample_means_results = central_limit_theorem(population_data, 1000, 10000)

# Visualize the results
visualize_results(sample_means_results)

# Take random sample and calculate the means
sample_means_results = central_limit_theorem(population_data, 10000, 10000)

# Visualize the results
visualize_results(sample_means_results)

