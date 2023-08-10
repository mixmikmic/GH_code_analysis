get_ipython().system('wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl')

try:
    import cPickle as pickle
except ImportError:
    # Python 3
    import pickle
    with open('vgg16.pkl', 'rb') as f:
        model = pickle.load(f, encoding='latin-1')
else:
    # Python 2
    with open('vgg16.pkl', 'rb') as f:
        model = pickle.load(f)

weights = model['param values']  # list of network weight tensors
classes = model['synset words']  # list of class names
mean_pixel = model['mean value']  # mean pixel value (in BGR)
del model

import os
# https://stackoverflow.com/questions/33988334/theano-config-directly-in-script
# disable GPU
os.environ["THEANO_FLAGS"] = "device=cpu,force_device=True"
import lasagne
import theano
assert theano.config.device == 'cpu'
assert theano.config.force_device == True
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
net['pool1'] = PoolLayer(net['conv1_2'], 2)
net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2_2'], 2)
net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
net['pool3'] = PoolLayer(net['conv3_3'], 2)
net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
net['pool4'] = PoolLayer(net['conv4_3'], 2)
net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5_3'], 2)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
net['prob'] = NonlinearityLayer(net['fc8'], softmax)

lasagne.layers.set_all_param_values(net['prob'], weights)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import urllib
import io
import skimage.transform
#from skimage.io import imread
import torch
# def prepare_image(url):
#     #ext = url.rsplit('.', 1)[1]
#     #img = plt.imread(io.BytesIO(urllib.request.urlopen(url).read()), ext)
#     img = imread(url)
#     # Resize so smallest dim = 256, preserving aspect ratio
#     h, w, _ = img.shape
#     if h < w:
#         img = skimage.transform.resize(img, (256, w*256//h), preserve_range=True)
#     else:
#         img = skimage.transform.resize(img, (h*256//w, 256), preserve_range=True)
#     # Central crop to 224x224
#     h, w, _ = img.shape
#     img = img[h//2-112:h//2+112, w//2-112:w//2+112]
#     # Remember this, it's a single RGB image suitable for plt.imshow()
#     img_original = img.astype('uint8')
#     # Shuffle axes from 01c to c01
#     img = img.transpose(2, 0, 1)
#     # Convert from RGB to BGR
#     img = img[::-1]
#     # Subtract mean pixel value
#     img = img - mean_pixel[:, np.newaxis, np.newaxis]
#     # Return the original and the prepared image (as a batch of a single item)
#     return img_original, lasagne.utils.floatX(img[np.newaxis])

# make sure preprocessing are the same.
from PIL import Image # so, this woorks better than skimage, as torchvision transforms work best with PIL and Tensor.
from torchvision import transforms

def prepare_image(url):
    img_to_use = Image.open(url)
    transform_1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
    ])

    # since it's 0-255 range.
    transform_2 = transforms.Compose([
        transforms.ToTensor(),
        # convert RGB to BGR
        # from <https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/util.py>
        transforms.Lambda(lambda x: torch.index_select(x, 0, torch.LongTensor([2, 1, 0]))),
        transforms.Lambda(lambda x: x*255),
        transforms.Normalize(mean = [103.939, 116.779, 123.68],
                              std = [ 1, 1, 1 ]),
    ])

    img_to_use_cropped = transform_1(img_to_use)
    img_to_use_cropped_tensor = transform_2(img_to_use_cropped)[np.newaxis]  # add first column for batching
    return np.array(img_to_use_cropped), img_to_use_cropped_tensor.numpy().copy()

import theano
import theano.tensor as T

def compile_saliency_function(net):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['fc8'], deterministic=True)
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])

def show_images(img_original, saliency, max_class, title):
    # get out the first map and class from the mini-batch
    print(saliency.dtype, img_original.dtype, max_class,
          saliency.min(), saliency.max(), saliency.mean(), saliency.std())
    saliency = saliency[0]
    max_class = max_class[0]
    # convert saliency from BGR to RGB, and from c01 to 01c
    saliency = saliency[::-1].transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    plt.show()

url = './4334173592_145856d89b.jpg'
img_original, img = prepare_image(url)
img.max()

saliency_fn = compile_saliency_function(net)
saliency, max_class = saliency_fn(img)
show_images(img_original, saliency, max_class, "default gradient")

relu = lasagne.nonlinearities.rectify
relu_layers = [layer for layer in lasagne.layers.get_all_layers(net['prob'])
               if getattr(layer, 'nonlinearity', None) is relu]

class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        print('init base')
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)

class GuidedBackprop(ModifiedBackprop):
    def __init__(self, nonlinearity):
        super().__init__(nonlinearity)
        print('init guided')
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
print(modded_relu.ops)
for layer in relu_layers:
    layer.nonlinearity = modded_relu

saliency_fn = compile_saliency_function(net)
print(img.min(), img.max())
saliency, max_class = saliency_fn(img)
show_images(img_original, saliency, max_class, "guided backprop")

# right.. you see? it's problematic.
class ZeilerBackprop(ModifiedBackprop):
    def __init__(self, nonlinearity):
        super().__init__(nonlinearity)
        print('init zeiler')
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
#         res_ = grd * (grd > 0).astype(inp.dtype)
#         print(type(res_), res_.std(), res_.max())
#         return (res_,)  # explicitly rectify
        #return (self.nonlinearity(grd),)  # use the given nonlinearity
        dtype = inp.dtype
        #return (grd,)
        #return (grd * (grd > 0).astype(dtype),)
        #print(grd.dtype, inp.dtype)
        #return (grd.astype(dtype),)
        #return (grd,)
        #return out_grads
        #return (grd * (grd > 0).astype(grd.dtype),)
        return (self.nonlinearity(grd),)

modded_relu_deconv = ZeilerBackprop(relu)
print(modded_relu_deconv.ops)
for layer in relu_layers:
    layer.nonlinearity = modded_relu_deconv
print(img.min(), img.max())
saliency_fn = compile_saliency_function(net)
saliency, max_class = saliency_fn(img)
show_images(img_original, saliency, max_class, "deconvnet")

url = './5595774449_b3f85b36ec.jpg'
img_original2, img2 = prepare_image(url)
for layer in relu_layers:
    layer.nonlinearity = relu
saliency_fn = compile_saliency_function(net)
show_images(img_original2, *saliency_fn(img2), title="default gradient")

modded_relu = GuidedBackprop(relu)
print(modded_relu.ops)
for layer in relu_layers:
    layer.nonlinearity = modded_relu
saliency_fn = compile_saliency_function(net)
show_images(img_original2, *saliency_fn(img2), title="guided backprop")

