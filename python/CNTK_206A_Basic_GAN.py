from IPython.display import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

import cntk as C
from cntk import Trainer
from cntk.device import try_set_default_device, gpu, cpu
from cntk.initializer import xavier
from cntk.io import (MinibatchSource, CTFDeserializer, StreamDef, StreamDefs,
                     INFINITELY_REPEAT)
from cntk.layers import Dense, default_options
from cntk.learners import (fsadagrad, UnitType, sgd, learning_rate_schedule,
                          momentum_as_time_constant_schedule)
from cntk.logging import ProgressPrinter

get_ipython().magic('matplotlib inline')

# Select the right target device when this notebook is being tested:
C.device.try_set_default_device(C.device.gpu(0))               

isFast = True 

# Ensure the training data is generated and available for this tutorial
# We search in two locations in the toolkit for the cached MNIST data set.

data_found = False
for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                 os.path.join("data", "MNIST")]:
    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    if os.path.isfile(train_file):
        data_found = True
        break
        
if not data_found:
    raise ValueError("Please generate the data by completing CNTK 103 Part A")
    
print("Data directory is {0}".format(data_dir))

def create_reader(path, is_training, input_dim, label_dim):
    deserializer = CTFDeserializer(
        filename = path,
        streams = StreamDefs(
            labels_unused = StreamDef(field = 'labels', shape = label_dim, is_sparse = False),
            features = StreamDef(field = 'features', shape = input_dim, is_sparse = False
            )
        )
    )
    return MinibatchSource(
        deserializers = deserializer,
        randomize = is_training,
        max_sweeps = INFINITELY_REPEAT if is_training else 1
    )

np.random.seed(123)
def noise_sample(num_samples):
    return np.random.uniform(
        low = -1.0,
        high = 1.0,
        size = [num_samples, g_input_dim]        
    ).astype(np.float32)

# Figure 1
Image(url="https://www.cntk.ai/jup/GAN_basic_flow.png")

# architectural parameters
g_input_dim = 100
g_hidden_dim = 128
g_output_dim = d_input_dim = 784
d_hidden_dim = 128
d_output_dim = 1

def generator(z):
    with default_options(init = xavier()):
        h1 = Dense(g_hidden_dim, activation = C.relu)(z)
        return Dense(g_output_dim, activation = C.tanh)(h1)

def discriminator(x):
    with default_options(init = xavier()):
        h1 = Dense(d_hidden_dim, activation = C.relu)(x)
        return Dense(d_output_dim, activation = C.sigmoid)(h1)

# training config
minibatch_size = 1024
num_minibatches = 300 if isFast else 40000
lr = 0.00005

# Figure 2
Image(url="https://www.cntk.ai/jup/GAN_goodfellow_NIPS2014.png", width = 500)

def build_graph(noise_shape, image_shape,
                G_progress_printer, D_progress_printer):
    input_dynamic_axes = [C.Axis.default_batch_axis()]
    Z = C.input(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input(image_shape, dynamic_axes=input_dynamic_axes)
    X_real_scaled = 2*(X_real / 255.0) - 1.0

    # Create the model function for the generator and discriminator models
    X_fake = generator(Z)
    D_real = discriminator(X_real_scaled)
    D_fake = D_real.clone(
        method = 'share',
        substitutions = {X_real_scaled.output: X_fake.output}
    )

    # Create loss functions and configure optimazation algorithms
    G_loss = 1.0 - C.log(D_fake)
    D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))

    G_learner = fsadagrad(
        parameters = X_fake.parameters,
        lr = learning_rate_schedule(lr, UnitType.sample),
        momentum = momentum_as_time_constant_schedule(700)
    )
    D_learner = fsadagrad(
        parameters = D_real.parameters,
        lr = learning_rate_schedule(lr, UnitType.sample),
        momentum = momentum_as_time_constant_schedule(700)
    )

    # Instantiate the trainers
    G_trainer = Trainer(
        X_fake,
        (G_loss, None),
        G_learner,
        G_progress_printer
    )
    D_trainer = Trainer(
        D_real,
        (D_loss, None),
        D_learner,
        D_progress_printer
    )

    return X_real, X_fake, Z, G_trainer, D_trainer

def train(reader_train):
    k = 2
    
    # print out loss for each model for upto 50 times
    print_frequency_mbsize = num_minibatches // 50
    pp_G = ProgressPrinter(print_frequency_mbsize)
    pp_D = ProgressPrinter(print_frequency_mbsize * k)

    X_real, X_fake, Z, G_trainer, D_trainer =         build_graph(g_input_dim, d_input_dim, pp_G, pp_D)
    
    input_map = {X_real: reader_train.streams.features}
    for train_step in range(num_minibatches):

        # train the discriminator model for k steps
        for gen_train_step in range(k):
            Z_data = noise_sample(minibatch_size)
            X_data = reader_train.next_minibatch(minibatch_size, input_map)
            if X_data[X_real].num_samples == Z_data.shape[0]:
                batch_inputs = {X_real: X_data[X_real].data, 
                                Z: Z_data}
                D_trainer.train_minibatch(batch_inputs)

        # train the generator model for a single step
        Z_data = noise_sample(minibatch_size)
        batch_inputs = {Z: Z_data}
        G_trainer.train_minibatch(batch_inputs)

        G_trainer_loss = G_trainer.previous_minibatch_loss_average

    return Z, X_fake, G_trainer_loss

reader_train = create_reader(train_file, True, d_input_dim, label_dim=10)

G_input, G_output, G_trainer_loss = train(reader_train)

# Print the generator loss 
print("Training loss of the generator is: {0:.2f}".format(G_trainer_loss))

def plot_images(images, subplot_shape):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(*subplot_shape)
    for image, ax in zip(images, axes.flatten()):
        ax.imshow(image.reshape(28, 28), vmin = 0, vmax = 1.0, cmap = 'gray')
        ax.axis('off')
    plt.show()
    
noise = noise_sample(36)
images = G_output.eval({G_input: noise})
plot_images(images, subplot_shape =[6, 6])

# Figure 3
Image(url="http://www.cntk.ai/jup/GAN_basic_slowmode.jpg")

