# global setup 
import sim_caffenet
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().magic('matplotlib inline')
import scipy.io as io
result_dir = './experiment_results/'
import collections
import cPickle as pickle

# run 1000 images from validation dataset
iterations = 20
batch_size = 50
LMDB_filename = '/media/jingyang/0E3519FE0E3519FE/ilsvrc12_val_lmdb/'

# run floating point CaffeNet (ground truth)
# so that we can get the range, and ground truth accuracy
floating_point_accuracy, floating_point_blobs_range, floating_point_weights_range,     floating_point_biases_range, kernels_name =     sim_caffenet.sim_floating_point_caffenet(LMDB_filename, batch_size=batch_size, 
                                             iterations=iterations, verbose=False)

# save the experiment results 
io.savemat(result_dir+'floating_point_acc.mat', dict(acc=floating_point_accuracy)) 

print 'Ground Truth (floating point): Top1 Acc: %f; Top5 Acc: %f' % (floating_point_accuracy[0].mean(), 
                                                                    floating_point_accuracy[1].mean())

uniform_bit_width = range(5, 17) # 5 - 16

# conservative floor strategy
round_method = 'FLOOR' # round method 
round_strategy = 'CONSERVATIVE' # round strategy
conservative_floor_uniform_accuracy = collections.OrderedDict()
for b in uniform_bit_width:
    print 'Simulate conservative + floor uniform bit %d...' % (b,)
    bit_width=dict(blobs=b, weights=b, biases=b) # uniform bit width
    conservative_floor_uniform_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
# save the experiment results 
io.savemat(result_dir+'conservative_floor_uniform_acc.mat', dict(bit_width=conservative_floor_uniform_accuracy.keys(),
                                                          acc=conservative_floor_uniform_accuracy.values()))

# conservative round strategy
round_method = 'ROUND' # round method 
round_strategy = 'CONSERVATIVE' # round strategy
conservative_round_uniform_accuracy = collections.OrderedDict()
for b in uniform_bit_width:
    print 'Simulate conservative + round uniform bit %d...' % (b,)
    bit_width=dict(blobs=b, weights=b, biases=b) # uniform bit width
    conservative_round_uniform_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
# save the experiment results 
io.savemat(result_dir+'conservative_round_uniform_acc.mat', dict(bit_width=conservative_round_uniform_accuracy.keys(),
                                                          acc=conservative_round_uniform_accuracy.values()))


# aggressive floor strategy
round_method = 'floor' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
aggressive_floor_uniform_accuracy = collections.OrderedDict()
for b in uniform_bit_width:
    print 'Simulate aggressive + floor uniform bit %d...' % (b,)
    bit_width=dict(blobs=b, weights=b, biases=b) # uniform bit width
    aggressive_floor_uniform_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
      
# save the experiment results 
io.savemat(result_dir+'aggressive_floor_uniform_acc.mat', dict(bit_width=aggressive_floor_uniform_accuracy.keys(),
                                                          acc=aggressive_floor_uniform_accuracy.values()))

# aggressive round strategy
round_method = 'ROUND' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
aggressive_round_uniform_accuracy = collections.OrderedDict()
for b in uniform_bit_width:
    print 'Simulate aggressive + round uniform bit %d...' % (b,)
    bit_width=dict(blobs=b, weights=b, biases=b) # uniform bit width
    aggressive_round_uniform_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
      
# save the experiment results 
io.savemat(result_dir+'aggressive_round_uniform_acc.mat', dict(bit_width=aggressive_round_uniform_accuracy.keys(),
                                                          acc=aggressive_round_uniform_accuracy.values()))

print 'Ground Truth (floating point): Top1 Acc: %f; Top5 Acc: %f' % (floating_point_accuracy[0].mean(), 
                                                                    floating_point_accuracy[1].mean())
plt.figure(figsize=(10, 5))
## Top1
plt.subplot(1, 2, 1)
plt.plot(uniform_bit_width, [conservative_floor_uniform_accuracy[b][0].mean() for b in uniform_bit_width], 'o-',
         linewidth=2, label='Conservative + Floor')
plt.plot(uniform_bit_width, [conservative_round_uniform_accuracy[b][0].mean() for b in uniform_bit_width], 'v-',
         linewidth=2, label='Conservative + Round')
plt.plot(uniform_bit_width, [aggressive_floor_uniform_accuracy[b][0].mean() for b in uniform_bit_width], 's-',
         linewidth=2, label='Aggressive + Floor')
plt.plot(uniform_bit_width, [aggressive_round_uniform_accuracy[b][0].mean() for b in uniform_bit_width], 'x-',
        linewidth=2, label='Aggressive + Round')
plt.title('Top-1 Accuracy with Uniform Fixed Point')
plt.xlabel('Bit Width')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')

## Top5
plt.subplot(1, 2, 2)
plt.plot(uniform_bit_width, [conservative_floor_uniform_accuracy[b][1].mean() for b in uniform_bit_width], 'o-',
         linewidth=2, label='Conservative + Floor')
plt.plot(uniform_bit_width, [conservative_round_uniform_accuracy[b][1].mean() for b in uniform_bit_width], 'v-',
         linewidth=2, label='Conservative + Round')
plt.plot(uniform_bit_width, [aggressive_floor_uniform_accuracy[b][1].mean() for b in uniform_bit_width], 's-',
         linewidth=2, label='Aggressive + Floor')
plt.plot(uniform_bit_width, [aggressive_round_uniform_accuracy[b][1].mean() for b in uniform_bit_width], 'x-',
        linewidth=2, label='Aggressive + Round')
plt.title('Top-5 Accuracy with Uniform Fixed Point')
plt.xlabel('Bit Width')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')

iterations = 20
batch_size = 50

blobs_bit_width = 16
kernels_bit_width = range(4, 17) # 4 ~ 16

# conservative floor strategy
round_method = 'FLOOR' # round method 
round_strategy = 'CONSERVATIVE' # round strategy
conservative_floor_kernels_accuracy = collections.OrderedDict()
for b in kernels_bit_width:
    print 'Simulate conservative floor kernels bit %d with blob bit %d...' % (b, blobs_bit_width)
    bit_width=dict(blobs=blobs_bit_width, weights=b, biases=b) # kernel bit width
    conservative_floor_kernels_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
# save the experiment results 
io.savemat(result_dir+'conservative_floor_kernels_acc.mat', dict(bit_width=conservative_floor_kernels_accuracy.keys(),
                                                          acc=conservative_floor_kernels_accuracy.values()))

# conservative round strategy
round_method = 'ROUND' # round method 
round_strategy = 'CONSERVATIVE' # round strategy
conservative_round_kernels_accuracy = collections.OrderedDict()
for b in kernels_bit_width:
    print 'Simulate conservative round kernels bit %d with blob bit %d...' % (b, blobs_bit_width)
    bit_width=dict(blobs=blobs_bit_width, weights=b, biases=b) # kernel bit width
    conservative_round_kernels_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)
# save the experiment results 
io.savemat(result_dir+'conservative_round_kernels_acc.mat', dict(bit_width=conservative_round_kernels_accuracy.keys(),
                                                          acc=conservative_round_kernels_accuracy.values()))
   
# aggressive floor strategy
round_method = 'FLOOR' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
aggressive_floor_kernels_accuracy = collections.OrderedDict()
for b in kernels_bit_width:
    print 'Simulate aggressive floor kernels bit %d with blob bit %d...' % (b, blobs_bit_width)
    bit_width=dict(blobs=blobs_bit_width, weights=b, biases=b) # kernel bit width
    aggressive_floor_kernels_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)    
    
# save the experiment results 
io.savemat(result_dir+'aggressive_floor_kernels_acc.mat', dict(bit_width=aggressive_floor_kernels_accuracy.keys(),
                                                          acc=aggressive_floor_kernels_accuracy.values()))    
    
# aggressive round strategy
round_method = 'ROUND' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
aggressive_round_kernels_accuracy = collections.OrderedDict()
for b in kernels_bit_width:
    print 'Simulate aggressive round kernels bit %d with blob bit %d...' % (b, blobs_bit_width)
    bit_width=dict(blobs=blobs_bit_width, weights=b, biases=b) # kernel bit width
    aggressive_round_kernels_accuracy[b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=False)    
    
# save the experiment results 
io.savemat(result_dir+'aggressive_round_kernels_acc.mat', dict(bit_width=aggressive_round_kernels_accuracy.keys(),
                                                          acc=aggressive_round_kernels_accuracy.values()))

print 'Ground Truth (floating point): Top1 Acc: %f; Top5 Acc: %f' % (floating_point_accuracy[0].mean(), 
                                                                    floating_point_accuracy[1].mean())
plt.figure(figsize=(10, 5))
## Top1
plt.subplot(1, 2, 1)
plt.plot(kernels_bit_width, [conservative_floor_kernels_accuracy[b][0].mean() for b in kernels_bit_width], 'o-',
         linewidth=2, label='Conservative + Floor')
plt.plot(kernels_bit_width, [conservative_round_kernels_accuracy[b][0].mean() for b in kernels_bit_width], '<-',
         linewidth=2, label='Conservative + Round')
plt.plot(kernels_bit_width, [aggressive_floor_kernels_accuracy[b][0].mean() for b in kernels_bit_width], 's-',
        linewidth=2, label='Aggressive + Floor')
plt.plot(kernels_bit_width, [aggressive_round_kernels_accuracy[b][0].mean() for b in kernels_bit_width], 'x-',
        linewidth=2, label='Aggressive + Round')
plt.title('Top-1 Accuracy with Uniform Fixed Point')
plt.xlabel('Bit Width')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')

## Top5
plt.subplot(1, 2, 2)
plt.plot(kernels_bit_width, [conservative_floor_kernels_accuracy[b][1].mean() for b in kernels_bit_width], 'o-',
         linewidth=2, label='Conservative + Floor')
plt.plot(kernels_bit_width, [conservative_round_kernels_accuracy[b][1].mean() for b in kernels_bit_width], '<-',
         linewidth=2, label='Conservative + Round')
plt.plot(kernels_bit_width, [aggressive_floor_kernels_accuracy[b][1].mean() for b in kernels_bit_width], 's-',
        linewidth=2, label='Aggressive + Floor')
plt.plot(kernels_bit_width, [aggressive_round_kernels_accuracy[b][1].mean() for b in kernels_bit_width], 'x-',
        linewidth=2, label='Aggressive + Round')
plt.title('Top-5 Accuracy with Uniform Fixed Point')
plt.xlabel('Bit Width')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', shadow=True, fontsize='x-large')

blobs_bit_width = 16
sensitive_kernel_bit_width = range(3, 17) # 3 ~ 16
default_kernel_bit_width = 16

# default kernel bit width
default_kernel = collections.OrderedDict();
for k in kernels_name:
    default_kernel[k] = default_kernel_bit_width

# optimal rounding strategy
round_method = 'ROUND' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
optimal_senstive_accuracy = collections.OrderedDict()
for k in kernels_name:
    optimal_senstive_accuracy[k] = collections.OrderedDict()
    for b in sensitive_kernel_bit_width:
        print 'Simulate aggressive kernels[%s] bit %d with blob bit %d...' % (k, b, blobs_bit_width)
        cur_kernel_bit_width = default_kernel.copy()
        cur_kernel_bit_width[k] = b
        print cur_kernel_bit_width
        bit_width=dict(blobs=blobs_bit_width, weights=cur_kernel_bit_width, biases=cur_kernel_bit_width) # kernel bit width
        optimal_senstive_accuracy[k][b] = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                             blobs_range=floating_point_blobs_range,                                              weights_range=floating_point_weights_range,                                               biases_range=floating_point_biases_range, batch_size=batch_size,                                               iterations=iterations, round_method=round_method,                                                 round_strategy=round_strategy, verbose=False)
        
# save experiements
pickle.dump(optimal_senstive_accuracy, open(result_dir+'optimal_sensitive_acc.p', 'wb'))
## load method
#load_optimal_senstive_accuracy = pickle.load(open(result_dir+'optimal_sensitive_acc.p', 'rb')) 

print 'Optimal sensitive analysis'
for k, v in optimal_senstive_accuracy.items():
    for bw, acc in v.items():
        print 'kernel[%s] @ bw = %d: top1 = %f; top5 = %f' %             (k, bw, acc.mean(axis=1)[0], acc.mean(axis=1)[1])



