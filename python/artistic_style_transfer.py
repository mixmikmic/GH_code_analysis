import numpy as np
import scipy as sp
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

def show_image(image, figsize=None, show_shape=False):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if show_shape:
        plt.title(image.shape)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

cat_img = plt.imread('../images/cat.835.jpg') # the image source is the reference [4]

show_image(cat_img, show_shape=True)

hokusai_img = plt.imread('../images/Tsunami_by_hokusai_19th_century.jpg')      

show_image(hokusai_img, show_shape=True)

TARGET_SIZE = cat_img.shape[:2]
hokusai_img = sp.misc.imresize(hokusai_img, TARGET_SIZE) # resize the style image to the content image size

show_image(hokusai_img, show_shape=True)

def preprocess(img):
    img = img.copy()                   # copy so that we don't mess the original data
    img = img.astype('float64')        # make sure it's float type
    img = np.expand_dims(img, axis=0)  # change 3-D to 4-D.  the first dimension is the record index
    return keras.applications.vgg16.preprocess_input(img)

def make_inputs(content_img, style_img):
    content_input   = K.constant(preprocess(content_img))
    style_input     = K.constant(preprocess(style_img))
    generated_input = K.placeholder(content_input.shape)  # use the same shape as the content input
    return content_input, style_input, generated_input

content_input, style_input, generated_input = make_inputs(cat_img, hokusai_img)

input_tensor = K.concatenate([content_input, style_input, generated_input], axis=0)

model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, include_top=False)

model.summary()

def calc_content_loss(layer_dict, content_layer_names):
    loss = 0
    for name in content_layer_names:
        layer = layer_dict[name]
        content_features   = layer.output[0, :, :, :]  # content features
        generated_features = layer.output[2, :, :, :]  # generated features
        loss += K.sum(K.square(generated_features - content_features)) # keep the similarity between them
    return loss / len(content_layer_names)

layer_dict = {layer.name:layer for layer in model.layers}

# use 'block5_conv2' as the representation of the content image
content_loss = calc_content_loss(layer_dict, ['block5_conv2'])

def gram_matrix(x):    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) # flatten per filter
    gram = K.dot(features, K.transpose(features)) # calculate the correlation between filters
    return gram

def get_style_loss(style_features, generated_features):
    S = gram_matrix(style_features)
    G = gram_matrix(generated_features)
    channels = 3
    size = TARGET_SIZE[0]*TARGET_SIZE[1]
    return K.sum(K.square(S - G)) / (4. * (channels**2) * (size**2))

def calc_style_loss(layer_dict, style_layer_names):
    loss = 0
    for name in style_layer_names:
        layer = layer_dict[name]
        style_features     = layer.output[1, :, :, :] # style features
        generated_features = layer.output[2, :, :, :] # generated features
        loss += get_style_loss(style_features, generated_features) 
    return loss / len(style_layer_names)

style_loss = calc_style_loss(
    layer_dict,
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'])

def calc_variation_loss(x):
    row_diff = K.square(x[:, :-1, :-1, :] - x[:, 1:,    :-1, :])
    col_diff = K.square(x[:, :-1, :-1, :] - x[:,  :-1, 1:,   :])
    return K.sum(K.pow(row_diff + col_diff, 1.25))

variation_loss = calc_variation_loss(generated_input)

loss = 0.8 * content_loss +        1.0 * style_loss   +        0.1 * variation_loss
        
grads = K.gradients(loss, generated_input)[0]

calculate = K.function([generated_input], [loss, grads])

generated_data = preprocess(cat_img)

for i in tqdm(range(10)):
    _, grads_value = calculate([generated_data])
    generated_data -= grads_value * 0.001
    

def deprocess(img):
    img = img.copy()                   # copy so that we don't mess the original data
    img = img[0]                       # take the 3-D image from the 4-D record    
    img[:, :, 0] += 103.939            # these are average color intensities used 
    img[:, :, 1] += 116.779            # by VGG16 which are subtracted from 
    img[:, :, 2] += 123.68             # the content image in the preprocessing
    img = img[:, :, ::-1]              # BGR -> RGB
    img = np.clip(img, 0, 255)         # clip the value within the image intensity range
    return img.astype('uint8')         # convert it to uin8 just like a normal image data

generated_image = deprocess(generated_data)

show_image(generated_image, figsize=(10,20))

def transfer_style(content_img, 
                   style_img,
                   content_layer_names, 
                   style_layer_names,
                   content_loss_ratio, 
                   style_loss_ratio, 
                   variation_loss_ratio,
                   start_img=None, 
                   steps=10,
                   learning_rate=0.001,
                   show_generated_image=True,
                   figsize=(10,20)):
    # clear the previous session if any
    K.clear_session()
    
    # by default start with the content image
    if start_img is None:
        start_img = content_img

    # prepare inputs and the model
    content_input, style_input, generated_input = make_inputs(content_img, style_img)
    input_tensor = K.concatenate([content_input, style_input, generated_input], axis=0)
    model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    
    # calculate various loss
    layer_dict = {layer.name:layer for layer in model.layers}
    content_loss = calc_content_loss(layer_dict, content_layer_names)
    style_loss = calc_style_loss(layer_dict, style_layer_names)
    variation_loss = calc_variation_loss(generated_input)
    
    # calculate the gradients
    loss = content_loss_ratio   * content_loss   +            style_loss_ratio     * style_loss     +            variation_loss_ratio * variation_loss

    grads = K.gradients(loss, generated_input)[0]
    calculate = K.function([generated_input], [loss, grads])

    # nudge the generated image to apply the style while keeping the content
    generated_data = preprocess(start_img)
    for i in tqdm(range(steps)):
        _, grads_value = calculate([generated_data])
        generated_data -= grads_value * learning_rate
        
    # reverse the preprocessing
    generated_img = deprocess(generated_data)
    
    if show_generated_image:
        show_image(generated_img, figsize=(10,20))
        
    return generated_img

transfer_style(
    cat_img, 
    hokusai_img,
    ['block5_conv2'], 
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'],
    content_loss_ratio=0.0, 
    style_loss_ratio=1.0, 
    variation_loss_ratio=0.0,
    steps=100,
    learning_rate=0.01);

transfer_style(
    cat_img, 
    hokusai_img,
    ['block5_conv2'], 
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'],
    content_loss_ratio=1.0, 
    style_loss_ratio=1.0, 
    variation_loss_ratio=0.0,
    steps=100,
    learning_rate=0.01);

transfer_style(
    cat_img, 
    hokusai_img,
    ['block5_conv2'], 
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'],
    content_loss_ratio=1.0, 
    style_loss_ratio=1.0, 
    variation_loss_ratio=0.7,               
    steps=100,
    learning_rate=0.01);

dancing_men_img = plt.imread('../images/dancing_men.png')      

dancing_men_img = sp.misc.imresize(dancing_men_img, TARGET_SIZE)

show_image(dancing_men_img, show_shape=True)

transfer_style(
    cat_img, 
    dancing_men_img,
    ['block5_conv1'], 
    ['block1_conv2',
     'block2_conv2',
     'block3_conv3'],
    content_loss_ratio=0.1, 
    style_loss_ratio=1.0, 
    variation_loss_ratio=0.1,
    steps=100,
    learning_rate=0.001);

random_img = np.random.uniform(0, 255, cat_img.shape)

show_image(random_img, show_shape=True)

transfer_style(
    cat_img, 
    dancing_men_img,
    ['block5_conv3'], 
    ['block1_conv2',
     'block2_conv2',
     'block3_conv3',
     'block4_conv3', 
     'block5_conv3'],
    content_loss_ratio=1.0, 
    style_loss_ratio=0.05, 
    variation_loss_ratio=0.01,
    start_img=random_img,
    steps=5000,
    learning_rate=0.03);

transfer_style(
    cat_img, 
    hokusai_img,
    ['block5_conv2'], 
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'],
    content_loss_ratio=1.0, 
    style_loss_ratio=0.05, 
    variation_loss_ratio=0.01,
    start_img=random_img,
    steps=5000,
    learning_rate=0.03);

