# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().magic('matplotlib inline')

# set display defaults
plt.rcParams['figure.figsize'] = (5, 5)          # medium images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # points to the root directory of caffe
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
from caffe import layers as L, params as P

# set CPU mode
caffe.set_mode_cpu()

# prepare the ILSVRC dataset
# path to the converted LMDB data 
# here we use the validation dataset
LMDB_filename = '/media/jingyang/0E3519FE0E3519FE/ilsvrc12_val_lmdb/'

# Load ImageNet labels to imagenet_labels
imagenet_label_file = caffe_root + '/data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
assert len(imagenet_labels) == 1000

# transformer
# Define the preprocessing step for CaffeNet
from scipy.ndimage import zoom
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = zoom(mu, (1, 227./256, 227./256))
transformer = caffe.io.Transformer({'data': (1, 3, 227, 227)})
transformer.set_transpose('data', (2, 0, 1)) # move channel to outmost dimension
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255.)
transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

# mean data: accompanied by the Caffe
mean_file = caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto'

# pre-trained model
caffenet_weights = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

batch_size = 10 # batch size for validation accuracy evaluation
iterations = 5 # number of iterations for validation accuracy evaluation 
                # the total number of images = batch_size * iterations

from caffenet_prototxt import caffenet

# filename for floating point caffenet
floating_point_caffenet_filename = 'floating_point_caffenet.prototxt'
# create input data layer for caffenet accuracy 
ilsvrc_data, ilsvrc_label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=LMDB_filename, ntop=2,
                    transform_param={'crop_size': 227, 'mirror': False, 'mean_file': mean_file})

# generate the prototxt file for CaffeNet definition
caffenet(ilsvrc_data, label=ilsvrc_label, train=False, num_classes=1000, classifier_name='fc8', 
         learn_all=False, filename=floating_point_caffenet_filename)

# create the floating point caffenet
floating_point_caffenet = None # workaround for sharing LMDB dataset
floating_point_caffenet = caffe.Net(floating_point_caffenet_filename, caffenet_weights, caffe.TEST)

# run the feedforward of floating point CaffeNet for validation accuracy evaluation
floating_point_acc = np.zeros((2, iterations)) # record the accuracy for each iteration
caffenet_blobs_range = {} # record the range of intermediate feature maps 
for k in floating_point_caffenet.blobs:
    if 'split' not in k:
        caffenet_blobs_range[k] = np.zeros((2, iterations)) # store min & max

for i in range(iterations):
    floating_point_caffenet.forward() # forward pass
    
    if i == 3:
        # show one batch result
        input_data = floating_point_caffenet.blobs['data'].data
        label_data = floating_point_caffenet.blobs['label'].data
        probs_data = floating_point_caffenet.blobs['probs'].data
        for idx in range(input_data.shape[0]): # over image idx
            top_5 = (-probs_data[idx]).argsort()[:5]
            print 'Top-5 Predicted: %s' % (str(top_5), )
            plt.figure(figsize=(3, 3))
            plt.imshow(transformer.deprocess('data', input_data[idx]))   
            plt.title('GT label: %d' % label_data[idx])
    
    for k in floating_point_caffenet.blobs:
        if 'split' not in k:
            caffenet_blobs_range[k][0, i] = floating_point_caffenet.blobs[k].data.min() # min value
            caffenet_blobs_range[k][1, i] = floating_point_caffenet.blobs[k].data.max() # max value
    floating_point_acc[0, i] = floating_point_caffenet.blobs['acc_top1'].data
    floating_point_acc[1, i] = floating_point_caffenet.blobs['acc_top5'].data
    print 'Batch %d: Top-1 accuracy: %f, Top-5 accuracy: %f' % (i, floating_point_acc[0, i],
                                                               floating_point_acc[1, i])

print 'Overall accuracy: Top-1: %f; Top-5: %f' % (floating_point_acc.mean(axis=1)[0],
                                                 floating_point_acc.mean(axis=1)[1])

blobs_range = {}
blobs_name = []
print 'Min & Max values at each layer'
for k in floating_point_caffenet.blobs:
    if 'split' not in k:
        smallest = caffenet_blobs_range[k][0].min()
        largest = caffenet_blobs_range[k][1].max()
        blobs_range[k] = [smallest, largest] # record the range 
        print '- layer[%s]: min %f; max %f' % (k, smallest, largest)
        blobs_name.append(k)

weights_range, biases_range = {}, {}
kernels_name = []
print 'Min & Max values of the parameters @ each layer'
for k, v in floating_point_caffenet.params.items():
    weights_range[k] = [v[0].data.min(), v[0].data.max()]
    biases_range[k] = [v[1].data.min(), v[1].data.max()]
    kernels_name.append(k)
    print '- kernels[%s]: weights [%f, %f]; biases [%f, %f]' % (k, v[0].data.min(), 
                                                               v[0].data.max(), v[1].data.min(),
                                                              v[1].data.max())
    
# remove the floating point caffent, otherwise sharing of LMDB is not allowed         
del floating_point_caffenet

# setup for the quantization layer
round_method = 'FLOOR' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
bit_width = 8

# here assume simple uniform bit width of the feature map of 
# CaffeNet
blobs_bit_width = {}
for k in blobs_name:
    blobs_bit_width[k] = bit_width
        
# here assume simple unifrom bit width of kernels (weights & biases) of CaffeNet
kernels_bit_width = {}
for k in kernels_name:
    kernels_bit_width[k] = bit_width

from caffenet_prototxt import convert_to_quantization_param, quantized_caffenet

# convert the quantization schemes to quantization parameter format
blobs_quantization_params = convert_to_quantization_param(blobs_bit_width, blobs_range,
                                    round_method=round_method, round_strategy=round_strategy)

# filename for quantized caffenet
quantized_caffenet_filename = 'quantized_caffenet.prototxt'

# generate the prototxt file for CaffeNet definition
quantized_caffenet(ilsvrc_data, blobs_quantization_params, label=ilsvrc_label, 
                   train=False, num_classes=1000, classifier_name='fc8', 
                   learn_all=False, filename=quantized_caffenet_filename)

# load the quantized caffenet
quantized_caffenet_net = None # workaround for sharing LMDB
quantized_caffenet_net = caffe.Net(quantized_caffenet_filename, caffenet_weights, caffe.TEST)

# import fixed_point function
from fixed_point import *

# quantize the kernels (weights + biases)
for layer_name, param in quantized_caffenet_net.params.items():
    # quantized weights: param[0] 
    WFixedPoint = FixedPoint(weights_range[layer_name], kernels_bit_width[layer_name], 
                             round_method=round_method, round_strategy=round_strategy)
    param[0].data[...] = WFixedPoint.quantize(param[0].data)   
    
    # quantized biases: param[1]
    BFixedPoint = FixedPoint(biases_range[layer_name], kernels_bit_width[layer_name],
                            round_method=round_method, round_strategy=round_strategy)
    param[1].data[...] = BFixedPoint.quantize(param[1].data)

quantized_acc = np.zeros((2, iterations))

print 'Quantization Scheme with Bit Width %d' % (bit_width, )

for i in range(iterations):
    quantized_caffenet_net.forward() # forward pass
    
    if i == 3:
        # show one batch result
        input_data = quantized_caffenet_net.blobs['data'].data
        label_data = quantized_caffenet_net.blobs['label'].data
        probs_data = quantized_caffenet_net.blobs['probs'].data
        for idx in range(input_data.shape[0]): # over image idx
            top_5 = (-probs_data[idx]).argsort()[:5]
            print 'Top-5 Predicted: %s' % (str(top_5), )
            plt.figure(figsize=(3, 3))
            plt.imshow(transformer.deprocess('data', input_data[idx]))   
            plt.title('GT label: %d' % label_data[idx])

    quantized_acc[0, i] = quantized_caffenet_net.blobs['acc_top1'].data
    quantized_acc[1, i] = quantized_caffenet_net.blobs['acc_top5'].data
    print 'Batch %d: Top-1 accuracy: %f, Top-5 accuracy: %f' % (i, quantized_acc[0, i],
                                                               quantized_acc[1, i])
    
print 'Overall accuracy: Top-1: %f; Top-5: %f' % (quantized_acc.mean(axis=1)[0],
                                                 quantized_acc.mean(axis=1)[1])

# remove quantized_caffenet_net
del quantized_caffenet_net

# re-set the parameters for running the exploration
iterations = 5
batch_size = 10
LMDB_filename = '/media/jingyang/0E3519FE0E3519FE/ilsvrc12_val_lmdb/'

import sim_caffenet

# running the floating point CaffeNet
floating_point_accuracy, floating_point_blobs_range, floating_point_weights_range,     floating_point_biases_range, kernels_name =     sim_caffenet.sim_floating_point_caffenet(LMDB_filename, batch_size=batch_size, 
                                             iterations=iterations, verbose=True)

iterations = 5
batch_size = 10
LMDB_filename = '/media/jingyang/0E3519FE0E3519FE/ilsvrc12_val_lmdb/'

round_method = 'FLOOR' # round method 
round_strategy = 'AGGRESSIVE' # round strategy
b = 8

bit_width = dict(blobs=b, weights=b, biases=b)
fixed_point_accuracy = sim_caffenet.sim_fixed_point_caffenet(LMDB_filename, bit_width=bit_width,                                         blobs_range=floating_point_blobs_range,                                          weights_range=floating_point_weights_range,                                           biases_range=floating_point_biases_range, batch_size=batch_size,                                           iterations=iterations, round_method=round_method,                                             round_strategy=round_strategy, verbose=True)



