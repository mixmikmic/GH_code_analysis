get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, sqrt, std, fabs

# stats60 specific
from code import roulette
from code.probability import Normal, SampleMean, Uniform
from code.utils import sample_density
figsize = (8,8)

get_ipython().run_cell_magic('R', '-h 800 -w 800', "PA.temp <- read.table('http://stats191.stanford.edu/data/paloaltoT.table', header=F, skip=2)\nplot(PA.temp[,3], xlab='Day', ylab='Average Max Temp (F)', pch=23, bg='orange')")

true_value = 3
SE_error_box = 2
normal_model = Normal(true_value, SE_error_box)
print normal_model.trial()
mean(normal_model.sample(2000)), std(normal_model.sample(2000))

get_ipython().run_cell_magic('capture', '', "normal_model_fig = plt.figure(figsize=figsize)\nax = sample_density(normal_model.sample(15000), bins=30, facecolor='orange')[0]\nax.set_title('True value = 3, normal error box SE = 2')")

normal_model_fig

sample_mean = SampleMean(normal_model, 3)
sample_mean.trial()

std(sample_mean.sample(5000)), 2 / sqrt(3)

get_ipython().run_cell_magic('capture', '', "sample_mean_fig = plt.figure(figsize=figsize)\nax = sample_density(sample_mean.sample(15000), bins=30, facecolor='orange')[0]\nax.set_title(r'True value = 3, sample size 3, sample mean SE = 2 / $\\sqrt{3}$')")

sample_mean_fig

other_model = Uniform(true_value, SE_error_box)
mean(other_model.sample(2000)), std(other_model.sample(2000))

get_ipython().run_cell_magic('capture', '', "other_model_fig = plt.figure(figsize=figsize)\nax = sample_density(other_model.sample(15000), bins=30, facecolor='orange')[0]\nax.set_title('True value = 3, error box SE = 2')")

other_model_fig

get_ipython().run_cell_magic('capture', '', "other_mean = SampleMean(other_model, 3)\nother_mean_fig = plt.figure(figsize=figsize)\nax = sample_density(other_mean.sample(15000), bins=30, facecolor='orange')[0]\nax.set_title(r'True value = 3, sample size 3, sample mean SE = 2 / $\\sqrt{3}$')")

other_mean_fig

get_ipython().run_cell_magic('capture', '', "other_mean = SampleMean(other_model, 20)\nother_mean_fig20 = plt.figure(figsize=figsize)\nax = sample_density(other_mean.sample(15000), bins=30, facecolor='orange')[0]\nax.set_title(r'True value = 3, sample size 3, sample mean SE = 2 / $\\sqrt{3}$')")

other_mean_fig20

def twoSD_proportion(sample_list):
    return mean([fabs(sample_list - mean(sample_list)) < 2 * std(sample_list)])
twoSD_proportion(normal_model.sample(500))

print 'sample size 5', mean([twoSD_proportion(normal_model.sample(5)) for _ in range(10000)])
print 'sample size 25', mean([twoSD_proportion(normal_model.sample(25)) for _ in range(10000)])

