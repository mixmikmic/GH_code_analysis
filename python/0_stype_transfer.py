import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
import PIL.Image
from IPython.display import Image, display
get_ipython().magic('matplotlib inline')

import vgg16
## download vgg model 
vgg16.maybe_download()

def load_image(filename,max_size=None):
    image = PIL.Image.open(filename)
    
    if max_size is not None:
        # Calculate the appropriate rescale-factor for ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)
        # Scale the image's height and width.
        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size,PIL.Image.LANCZOS)
    ## convert to numpy floating-pint array 
    return np.float32(image)

# Save an image as a jpeg-file. The image is given as a numpy array 
# with pixel-values between 0 and 255.
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    # Convert to bytes.
    image = image.astype(np.uint8)
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

# Plot a image 
def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    # Convert pixels to bytes.
    image = image.astype(np.uint8)
    # Convert to a PIL-image and display it.
    display(PIL.Image.fromarray(image))

###############################################################
### This function plots the content-, mixed- and style-images.
###############################################################
def plot_images(content_image, style_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # Use interpolation to smooth pixels?
    smooth = True
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")
    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")
    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")
    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

## mean squared error
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

#step 3 - content image is 3d numpy array, indices for the layers
#we want to use for content loss
#you should expirment what looks good for different layers
#there is not one best layer, we haven't found a way to minimize
#loss for beauty. how to quantify?

def create_content_loss(session,model,content_image,layer_ids):
    '''
    Create the loss-function for the content-image.
    
    Parameters:
    session: An open TensorFlow session for running the model's graph.
    model: The model, e.g. an instance of the VGG16-class.
    content_image: Numpy float array with the content-image.
    layer_ids: List of integer id's for the layers to use in the model.
    '''
    # Create a feed-dict with the content-image.
    # input a 3 d image, return a feed_dict = {self.tensor_name_input_image: image}
    # image is a 4 d array
    feed_dict = model.create_feed_dict(image=content_image)
    # Get references to the tensors for the given layers.
    # collection of filters---------------------------- is this weights???
    layers = model.get_layer_tensors(layer_ids)
    # Calculate the output values of those layers when
    # feeding the content-image to the model.
    values = session.run(layers,feed_dict=feed_dict)
    with model.graph.as_default():
        # Initialize an empty list of loss-functions.
        #because we are calculating losses per layer
        layer_losses=[]
        # For each layer and its corresponding values
        # for the content-image.
        for value,layer in zip(values,layers):
            # These are the values that are calculated
            # for this layer in the model when inputting
            # the content-image. Wrap it to ensure it
            # is a const - although this may be done
            # automatically by TensorFlow.
            value_const = tf.constant(value)
            ## take the mean square error of these two
            ## it does look strange in the first place, isn't values_const and layer the same thing?
            ## actually they are not, value is the activation layer of the content image 
            ## while runing the optimization, we are feeding through the result image, mixed_img
            ## so the loss will be the difference of mixed_image and the content_image
            loss = mean_squared_error(layer,value_const)
            # list of loss-functions
            layer_losses.append(loss)
            
        # The combined loss for all layers is just the average.
        total_loss = tf.reduce_mean(layer_losses)
    
    return total_loss

#The Gram matrix, defined as https://i.stack.imgur.com/tU9ow.png 
#is used to 
#measure the correlation between channels after flattening the 
#filter images into vectors

#Gatys when asked why gram matrix at a talk was that the 
#Gram matrix encodes second 
#order statistics of the set of filters.
#it sort of mushes up all the features at a given layer, 
#tossing spatial information in favor of a measure of 
#how the different features are correlated 
def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])  # for convolutions layers, it is assuming you use batches, 
                                 # that is why it is 4 dimentiosn
    matrix = tf.reshape(tensor,shape=[-1,num_channels])  ## flaten each layer
    gram = tf.matmul(tf.transpose(matrix),matrix)        ## this is the variance co-variance matrix
    
    return gram

def create_style_loss(session,model,style_image,layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)
    layers = model.get_layer_tensors(layer_ids)
    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        values = session.run(gram_layers, feed_dict=feed_dict)
        layer_losses=[]
        for value,gram_layer in zip(values,gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer,value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss

#shifts input image by 1 pixel on x and y axis 
#calculate difference between shifted and original image
#absolute value to make positive
#calculate sum of pixels in those images
#helps suppress noise in mixed image we are generating
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) +            tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss

def style_transfer(content_image,style_image,
                  content_layer_ids,style_layer_ids,
                  weight_content=1.5,weight_style=10.0,
                  weight_denoise=0.3,
                  num_iterations=120,step_size=10.0):
    
    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph=model.graph)
    
    # Print the names of the content-layers.
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()
    # Print the names of the style-layers.
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()
    
    # Create the loss-function for the content-layers and -image.
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)
    
    # Create the loss-function for the style-layers and -image.
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)  
    
    # Create the loss-function for the denoising of the mixed-image.
    loss_denoise = create_denoise_loss(model)
    
    # Create TensorFlow variables for adjusting the values of
    # the loss-functions. This is explained below.
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')
    
    session.run([adj_content.initializer,
             adj_style.initializer,
             adj_denoise.initializer])
    
    # Create TensorFlow operations for updating the adjustment values.
    # These are basically just the reciprocal values of the
    # loss-functions, with a small value 1e-10 added to avoid the
    # possibility of division by zero.
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))
    
    # This is the weighted loss-function that we will minimize
    # below in order to generate the mixed-image.
    # Because we multiply the loss-values with their reciprocal
    # adjustment values, we can use relative weights for the
    # loss-functions that are easier to select, as they are
    # independent of the exact choice of style- and content-layers.
    loss_combined = weight_content * adj_content * loss_content +                     weight_style * adj_style * loss_style +                     weight_denoise * adj_denoise * loss_denoise
            
    # Use TensorFlow to get the mathematical function for the
    # gradient of the combined loss-function with regard to
    # the input image. (mixed)
    gradient = tf.gradients(loss_combined, model.input) ## delta change in terms of input
                                                        ## back propogation do it in terms 
                                                        ## of wieghts, here,we do it in terms
                                                        ## of input 
    
    # List of tensors that we will run in each optimization iteration.
    run_list = [gradient, update_adj_content, update_adj_style,                 update_adj_denoise]
    
    # The mixed-image is initialized with random noise.
    # It is the same size as the content-image.
    # where we first init it
    mixed_image = np.random.rand(*content_image.shape) + 128
    
    for i in range(num_iterations):
        #Create a feed-dict with the mixed-image.
        feed_dict = model.create_feed_dict(image=mixed_image)
        # run get all the values
        grad, adj_content_val, adj_style_val, adj_denoise_val         = session.run(run_list, feed_dict=feed_dict)     
        
        ## squeeze empty dimentions, because cnn need at least 4 dimentions
        grad = np.squeeze(grad)   ## it should be 3 dimentions here 
        
        ## this is like learning rate, how many standard deviations we want to move
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        
        # Update the image by following the gradient.
        # gradient descent
        mixed_image -= grad * step_size_scaled
        
        # Ensure the image has valid pixel-values between 0 and 255.
        # Given an interval, values outside the interval are clipped 
        # to the interval edges.
        mixed_image = np.clip(mixed_image, 0.0, 255.0)
        
        # Print a little progress-indicator.
        # print(". ", end="")

        # Display status once every 10 iterations, and the last.
        if (i % 300 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            # Print adjustment weights for loss-functions.
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            #in larger resolution
            # Plot the content-, style- and mixed-images.
            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)
    
    print()
    print("Final image:")
    plot_image_big(mixed_image)

    # Close the TensorFlow session to release its resources.
    session.close()
    
    # Return the mixed-image.
    return mixed_image

content_filename = 'images/content5.jpg'
content_image = load_image(content_filename, max_size=None)

style_filename = 'images/style3.jpg'
style_image = load_image(style_filename, max_size=300)

content_layer_ids = [3]

# The VGG16-model has 13 convolutional layers.
# This selects all those layers as the style-layers.
# This is somewhat slow to optimize.
style_layer_ids = list(range(5,13))

# You can also select a sub-set of the layers, e.g. like this:
# style_layer_ids = [6,7,8,9]

get_ipython().run_cell_magic('time', '', 'img = style_transfer(content_image=content_image,\n                     style_image=style_image,\n                     content_layer_ids=content_layer_ids,\n                     style_layer_ids=style_layer_ids,\n                     weight_content=10.0,\n                     weight_style=10.0,\n                     weight_denoise=1.0,\n                     num_iterations=600,\n                     step_size=10.0)')



