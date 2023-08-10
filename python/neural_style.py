get_ipython().magic('matplotlib inline')

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc     import imsave
from keras          import metrics
from vgg16_avg      import VGG16_Avg
from PIL            import Image

from keras.models   import Model

import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt
import scipy.ndimage

def limit_mem():
    cfg                          = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config = cfg))

limit_mem()

content_fn = '../data/pictures/dog_02_small.jpg'
style_fn   = '../data/pictures/van_gogh.jpg'

content       = Image.open(content_fn)
content_array = np.array(content)
plt.imshow(content_array)

style       = Image.open(style_fn)
style_array = scipy.misc.imresize(style, content.size[::-1])
plt.imshow(style_array)

imagenet_component_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32)

def imagenet_preproc(image_array):
    return (image_array - imagenet_component_mean)[:, :, :, ::-1]

def undo_imagenet_preproc(image_array, shape):
    return np.clip(image_array.reshape(shape)[:, :, :, ::-1] + imagenet_component_mean, 0, 255)

def plot_array(array):
    plt.imshow(undo_imagenet_preproc(array, array.shape)[0].astype('uint8'))

content_batch = np.expand_dims(content_array, 0)
content_shape = content_batch.shape
style_batch   = np.expand_dims(style_array, 0) 
style_shape   = style_batch.shape

content_batch = imagenet_preproc(content_batch)
style_batch   = imagenet_preproc(style_batch)

model = VGG16_Avg(include_top = False)

for layer in model.layers:
    print(layer.name)

content_layer = model.get_layer('block5_conv1').output

layer_model    = Model(model.input, content_layer)
content_target = K.variable(layer_model.predict(content_batch))

class Evaluator(object):
    def __init__(self, function, shape):
        self.function = function
        self.shape    = shape
        
    def loss(self, x):
        loss_, self.gradient_values = self.function([x.reshape(self.shape)])
        
        return loss_.astype(np.float64)
    
    def gradients(self, x):
        return self.gradient_values.flatten().astype(np.float64)

loss      = metrics.mse(content_layer, content_target)
gradients = K.gradients(loss, model.input)
fn        = K.function([model.input], [loss] + gradients)
evaluator = Evaluator(fn, content_shape)

def solve_image(evaluator, iteration_number, x):
    shape = x.shape
    for i in range(iteration_number):
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime = evaluator.gradients, maxfun = 20)
        
        x = np.clip(x, -127, 127)
        print('Current loss value:', min_val)
        imsave('../data/pictures/res_at_iteration_%02d.png' % i, undo_imagenet_preproc(x.copy(), shape)[0])
    return x

rand_image = lambda shape: np.random.uniform(-2.5, 2.5, shape) / 100
x          = rand_image(content_shape)
plt.imshow(x[0])

iteration_number = 10

x = solve_image(evaluator, iteration_number, x)

from IPython.display import HTML
from matplotlib      import animation, rc

fig, ax = plt.subplots()

def animate(i):
    ax.imshow(Image.open('../data/pictures/res_at_iteration_%02d.png' % i))

anim = animation.FuncAnimation(fig, animate, frames = 10, interval = 200)
HTML(anim.to_html5_video())

model   = VGG16_Avg(include_top = False, input_shape = style_shape[1:])
outputs = {layer.name : layer.output for layer in model.layers} 

outputs

style_layers = [outputs['block%d_conv1' % block_number] for block_number in range(1, 3)]
style_layers

layers_model  = Model(model.input, style_layers)
style_targets = [K.variable(output) for output in layers_model.predict(style_batch)] 

def gram_matrix(x):
    # We want each row to be a channel and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

def style_loss(x, target):
    return metrics.mse(gram_matrix(x), gram_matrix(target))

loss      = sum(style_loss(l1[0], l2[0]) for l1, l2 in zip(style_layers, style_targets))
gradients = K.gradients(loss, model.input)
style_fn  = K.function([model.input], [loss] + gradients)
evaluator = Evaluator(style_fn, style_shape)

rand_image = lambda shape: np.random.uniform(-2.5, 2.5, shape) / 1
x          = rand_image(style_shape)
x          = scipy.ndimage.filters.gaussian_filter(x, [0, 2, 2, 0])
plt.imshow(x[0])

iteration_number = 10

x = solve_image(evaluator, iteration_number, x)

fig, ax = plt.subplots()

def animate(i):
    ax.imshow(Image.open('../data/pictures/res_at_iteration_%02d.png' % i))

anim = animation.FuncAnimation(fig, animate, frames = 10, interval = 200)
HTML(anim.to_html5_video())

style_layers  = [outputs['block%d_conv2' % output] for output in range(1, 6)]
content_name  = 'block4_conv2'
content_layer = outputs[content_name]

style_model   = Model(model.input, style_layers)
style_targets = [K.variable(o) for o in style_model.predict(style_batch)]

content_model  = Model(model.input, content_layer)
content_target = K.variable(content_model.predict(content_batch))

style_weights = [0.05, 0.2, 0.2, 0.25, 0.3]

loss        = sum(style_loss(l1[0], l2[0]) * w for l1, l2, w in zip(style_layers, style_targets, style_weights))
loss        = loss + metrics.mse(content_layer, content_target) / 10
gradients   = K.gradients(loss, model.input)
transfer_fn = K.function([model.input], [loss] + gradients)

evaluator = Evaluator(transfer_fn, content_shape)

iterations = 50
x          = rand_image(content_shape)

x = solve_image(evaluator, iterations, x)

Image.open('../data/pictures/res_at_iteration_49.png')

