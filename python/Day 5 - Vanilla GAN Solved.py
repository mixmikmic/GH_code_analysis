get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, util

data_folder = 'data'; dataset = 'mnist'  # the folder in which the dataset is going to be stored

download_folder = util.download_mnist(data_folder, dataset)
images, labels = util.load_mnist(download_folder)

print("Folder:", download_folder)
print("Image shape:", images.shape) # greyscale, so the last dimension (color channel) = 1
print("Label shape:", labels.shape) # one-hot encoded

show_n_images = 25
sample_images, mode = util.get_sample_images(images, n=show_n_images)
mnist_sample = util.images_square_grid(sample_images, mode)
plt.imshow(mnist_sample, cmap='gray')

sample = images[3]*50 # 
sample = sample.reshape((28, 28))
print(np.array2string(sample.astype(int), max_line_width=100, separator=',', precision=0))

plt.imshow(sample, cmap='gray')

# the mnist dataset is stored in the variable 'images', and the labels are stored in 'labels'
images = images.reshape(-1, 28*28) # 70000 x 784
print (images.shape, labels.shape) 

mnist = util.Dataset(images, labels)
print ("Number of samples:", mnist.n)

class Generator:
    """The generator network
    
    the generator network takes as input a vector z of dimension input_dim, and transforms it 
    to a vector of size output_dim. The network has one hidden layer of size hidden_dim.
    
    We will define the following methods: 
    
    __init__: initializes all variables by using tf.get_variable(...) 
                and stores them to the class, as well a list in self.theta
    forward: defines the forward pass of the network - how do the variables
                interact with respect to the inputs
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Constructor for the generator network. In the constructor, we will
        just initialize all the variables in the network.
        
        Args:
            input_dim: The dimension of the input data vector (z).
            hidden_dim: The dimension of the hidden layer of the neural network (h)
            output_dim: The dimension of the output layer (equivalent to the size of the image)            
            
        """
        
        with tf.variable_scope("generator"):
            self.W1 = tf.get_variable(name="W1", 
                                      shape=[input_dim, hidden_dim], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable(name="b1", 
                                      shape=[hidden_dim], 
                                      initializer=tf.zeros_initializer())
            
            self.W2 = tf.get_variable(name="W2", 
                                      shape=[hidden_dim, output_dim], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable(name="b2", 
                                      shape=[output_dim], 
                                      initializer=tf.zeros_initializer())

            self.theta = [self.W1, self.W2, self.b1, self.b2]
    
    def forward(self, z):
        """The forward pass of the network -- here we will define the logic of how we combine
        the variables through multiplication and activation functions in order to get the
        output.
        
        """
        h1 = tf.nn.relu(tf.matmul(z, self.W1) + self.b1)
        
        log_prob = tf.matmul(h1, self.W2) + self.b2
        prob = tf.nn.sigmoid(log_prob)

        return prob

class Discriminator:
    """The discriminator network
    
    the discriminator network takes as input a vector x of dimension input_dim, and transforms it 
    to a vector of size output_dim. The network has one hidden layer of size hidden_dim.
    
    You will define the following methods: 
    
    __init__: initializes all variables by using tf.get_variable(...) 
                and stores them to the class, as well a list in self.theta
    forward: defines the forward pass of the network - how do the variables
                interact with respect to the inputs
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        with tf.variable_scope("discriminator"):
            
            self.W1 = tf.get_variable(name="W1", 
                                      shape=[input_dim, hidden_dim], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable(name="b1", shape=[hidden_dim], 
                                      initializer=tf.zeros_initializer())
            
            self.W2 = tf.get_variable(name="W2", 
                                      shape=[hidden_dim, output_dim], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable(name="b2", 
                                      shape=[output_dim], 
                                      initializer=tf.zeros_initializer())

            self.theta = [self.W1, self.W2, self.b1, self.b2]
    
    def forward(self, x):
        """The forward pass of the network -- here we will define the logic of how we combine
        the variables through multiplication and activation functions in order to get the
        output.
        
        """
        h1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        logit = tf.matmul(h1, self.W2) + self.b2
        prob = tf.nn.sigmoid(logit)

        return prob, logit

image_dim = 784 # The dimension of the input image vector to the discrminator
discriminator_hidden_dim = 128 # The dimension of the hidden layer of the discriminator
discriminator_output_dim = 1 # The dimension of the output layer of the discriminator 

random_sample_dim = 100 # The dimension of the random noise vector z
generator_hidden_dim = 128 # The dimension of the hidden layer of the generator
generator_output_dim = 784 # The dimension of the output layer of the generator

d = Discriminator(image_dim, discriminator_hidden_dim, discriminator_output_dim)
for param in d.theta:
    print (param)

g = Generator(random_sample_dim, generator_hidden_dim, generator_output_dim)
for param in g.theta:
    print (param)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

plt.imshow(sample_Z(16, 100), cmap='gray')

def gan_model_loss(X, Z, discriminator, generator):
    G_sample = g.forward(Z)
    D_real, D_logit_real = d.forward(X)
    D_fake, D_logit_fake = d.forward(G_sample)

    D_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

    D_loss = D_loss_real + D_loss_fake

    G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    return G_sample, D_loss, G_loss
    

X = tf.placeholder(tf.float32, name="input", shape=[None, image_dim])
Z = tf.placeholder(tf.float32, name="latent_sample", shape=[None, random_sample_dim])

G_sample, D_loss, G_loss = gan_model_loss(X, Z, d, g)

with tf.variable_scope('optim'):
    D_solver = tf.train.AdamOptimizer(name='discriminator').minimize(D_loss, var_list=d.theta)
    G_solver = tf.train.AdamOptimizer(name='generator').minimize(G_loss, var_list=g.theta)

saver = tf.train.Saver()

# Some runtime parameters predefined for you
minibatch_size = 128 # The size of the minibatch

num_epoch = 500 # For how many epochs do we run the training
plot_every_epochs = 5 # After this many epochs we will save & display samples of generated images 
print_every_batches = 1000 # After this many minibatches we will print the losses

restore = False
checkpoint = 'fc_2layer_e100_2.170.ckpt'
model = 'gan'
model_save_folder = os.path.join('data', 'chkp', model)
print ("Model checkpoints will be saved to:", model_save_folder)
image_save_folder = os.path.join('data', 'model_output', model)
print ("Image samples will be saved to:", image_save_folder)

minibatch_counter = 0
epoch_counter = 0

d_losses = []
g_losses = []

with tf.device("/gpu:0"), tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if restore:
        saver.restore(sess, os.path.join(model_save_folder, checkpoint))
        print("Restored model:", checkpoint, "from:", model_save_folder)
                      
    while epoch_counter < num_epoch:
            
        new_epoch, X_mb = mnist.next_batch(minibatch_size)

        _, D_loss_curr = sess.run([D_solver, D_loss], 
                                  feed_dict={
                                      X: X_mb, 
                                      Z: sample_Z(minibatch_size, random_sample_dim)
                                    })
                      
        _, G_loss_curr = sess.run([G_solver, G_loss], 
                                  feed_dict={
                                      Z: sample_Z(minibatch_size, random_sample_dim)
                                  })

        # Plotting and saving images and the model
        if new_epoch and epoch_counter % plot_every_epochs == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, random_sample_dim)})

            fig = util.plot(samples)
            figname = '{}.png'.format(str(minibatch_counter).zfill(3))
            plt.savefig(os.path.join(image_save_folder, figname), bbox_inches='tight')
            plt.show()
            plt.close(fig)
            
            im = util.plot_single(samples[0], epoch_counter)
            plt.savefig(os.path.join(image_save_folder, 'single_' + figname), bbox_inches='tight')
            plt.show()
            
            chkpname = "fc_2layer_e{}_{:.3f}.ckpt".format(epoch_counter, G_loss_curr)
            saver.save(sess, os.path.join(model_save_folder, chkpname))

        # Printing runtime statistics
        if minibatch_counter % print_every_batches == 0:
            print('Epoch: {}/{}'.format(epoch_counter, num_epoch))
            print('Iter: {}/{}'.format(mnist.position_in_epoch, mnist.n))
            print('Discriminator loss: {:.4}'. format(D_loss_curr))
            print('Generator loss: {:.4}'.format(G_loss_curr))
            print()
        
        # Bookkeeping
        minibatch_counter += 1
        if new_epoch:
            epoch_counter += 1
        
        d_losses.append(D_loss_curr)
        g_losses.append(G_loss_curr)
        
    chkpname = "fc_2layer_e{}_{:.3f}.ckpt".format(epoch_counter, G_loss_curr)
    saver.save(sess, os.path.join(model_save_folder, chkpname))

disc_line, = plt.plot(range(len(d_losses[:10000])), d_losses[:10000], c='b', label="Discriminator loss")
gen_line, = plt.plot(range(len(d_losses[:10000])), g_losses[:10000], c='r', label="Generator loss")
plt.legend([disc_line, gen_line], ["Discriminator loss", "Generator loss"])



