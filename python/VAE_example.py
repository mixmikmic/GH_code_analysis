import mxnet as mx
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mnist = mx.test_utils.get_mnist()
image = np.reshape(mnist['train_data'],(60000,28*28))
label = image
image_test = np.reshape(mnist['test_data'],(10000,28*28))
label_test = image_test
[N,features] = np.shape(image)          #number of examples and features

f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,  sharex='col', sharey='row',figsize=(12,3))
ax1.imshow(np.reshape(image[0,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax2.imshow(np.reshape(image[1,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax3.imshow(np.reshape(image[2,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax4.imshow(np.reshape(image[3,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
plt.show()

model_prefix = None

batch_size = 100
nd_iter = mx.io.NDArrayIter(data={'data':image},label={'loss_label':label},
                            batch_size = batch_size)
nd_iter_test = mx.io.NDArrayIter(data={'data':image_test},label={'loss_label':label_test},
                            batch_size = batch_size)

## define data and loss labels as symbols 
data = mx.sym.var('data')
loss_label = mx.sym.var('loss_label')

## define fully connected and activation layers for the encoder, where we used tanh activation function.
encoder_h  = mx.sym.FullyConnected(data=data, name="encoder_h",num_hidden=400)
act_h = mx.sym.Activation(data=encoder_h, act_type="tanh",name="activation_h")

## define mu and log variance which are the fully connected layers of the previous activation layer
mu  = mx.sym.FullyConnected(data=act_h, name="mu",num_hidden = 5)
logvar  = mx.sym.FullyConnected(data=act_h, name="logvar",num_hidden = 5)

## sample the latent variables z according to Normal(mu,var)
z = mu + np.multiply(mx.symbol.exp(0.5*logvar),mx.symbol.random_normal(loc=0, scale=1,shape=np.shape(logvar.get_internals()["logvar_output"])))

# define fully connected and tanh activation layers for the decoder
decoder_z = mx.sym.FullyConnected(data=z, name="decoder_z",num_hidden=400)
act_z = mx.sym.Activation(data=decoder_z, act_type="tanh",name="activation_z")

# define the output layer with sigmoid activation function, where the dimension is equal to the input dimension
decoder_x = mx.sym.FullyConnected(data=act_z, name="decoder_x",num_hidden=features)
y = mx.sym.Activation(data=decoder_x, act_type="sigmoid",name='activation_x')

# define the objective loss function that needs to be minimized
KL = 0.5*mx.symbol.sum(1+logvar-pow( mu,2)-mx.symbol.exp(logvar),axis=1)
loss = -mx.symbol.sum(mx.symbol.broadcast_mul(loss_label,mx.symbol.log(y)) + mx.symbol.broadcast_mul(1-loss_label,mx.symbol.log(1-y)),axis=1)-KL
output = mx.symbol.MakeLoss(sum(loss),name='loss')

# set up the log
nd_iter.reset()
logging.getLogger().setLevel(logging.DEBUG)  

#define function to trave back training loss
def log_to_list(period, lst):
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0:
            name, value = param.eval_metric.get()
            lst.append(value)
    return _callback

# define the model
model = mx.mod.Module(
    symbol = output ,
    data_names=['data'],
    label_names = ['loss_label'])

# training the model, save training loss as a list.
training_loss=list()

# initilize the parameters for training using Normal.
init = mx.init.Normal(0.01)
model.fit(nd_iter,  # train data
              initializer=init,
              #if eval_data is supplied, test loss will also be reported
              #eval_data = nd_iter_test,
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':1e-3,'wd':1e-2},  
              epoch_end_callback  = None if model_prefix==None else mx.callback.do_checkpoint(model_prefix, 1),   #save parameters for each epoch if model_prefix is supplied
              batch_end_callback = log_to_list(N/batch_size,training_loss), 
              num_epoch=100,
              eval_metric = 'Loss')

ELBO = [-training_loss[i] for i in range(len(training_loss))]
plt.plot(ELBO)
plt.ylabel('ELBO');plt.xlabel('epoch');plt.title("training curve for mini batches")
plt.show()

arg_params = model.get_params()[0]

#if saved the parameters, can load them at e.g. 100th epoch
#sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 100)
#assert sym.tojson() == output.tojson()

e = y.bind(mx.cpu(), {'data': nd_iter_test.data[0][1],
                     'encoder_h_weight': arg_params['encoder_h_weight'],
                     'encoder_h_bias': arg_params['encoder_h_bias'],
                     'mu_weight': arg_params['mu_weight'],
                     'mu_bias': arg_params['mu_bias'],
                     'logvar_weight':arg_params['logvar_weight'],
                     'logvar_bias':arg_params['logvar_bias'],
                     'decoder_z_weight':arg_params['decoder_z_weight'],
                     'decoder_z_bias':arg_params['decoder_z_bias'],
                     'decoder_x_weight':arg_params['decoder_x_weight'],
                     'decoder_x_bias':arg_params['decoder_x_bias'],                
                     'loss_label':label})

x_fit = e.forward()
x_construction = x_fit[0].asnumpy()

# learning images on the test set
f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4,  sharex='col', sharey='row',figsize=(12,3))
ax1.imshow(np.reshape(image_test[0,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax1.set_title('True image')
ax2.imshow(np.reshape(x_construction[0,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax2.set_title('Learned image')
ax3.imshow(np.reshape(x_construction[999,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax3.set_title('Learned image')
ax4.imshow(np.reshape(x_construction[9999,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax4.set_title('Learned image')
plt.show()

#calculate the ELBO which is minus the loss for test set
metric = mx.metric.Loss()
model.score(nd_iter_test, metric)

from VAE import VAE

# can initilize weights and biases with the learned parameters 
#init = mx.initializer.Load(params)

# call the VAE , output model contains the learned model and training loss
out = VAE(n_latent=2,x_train=image,x_valid=None,num_epoch=200) 

# encode test images to obtain mu and logvar which are used for sampling
[mu,logvar] = VAE.encoder(out,image_test)
#sample in the latent space
z = VAE.sampler(mu,logvar)
# decode from the latent space to obtain reconstructed images
x_construction = VAE.decoder(out,z)

f, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4,  sharex='col', sharey='row',figsize=(12,3))
ax1.imshow(np.reshape(image_test[0,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax1.set_title('True image')
ax2.imshow(np.reshape(x_construction[0,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax2.set_title('Learned image')
ax3.imshow(np.reshape(x_construction[999,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax3.set_title('Learned image')
ax4.imshow(np.reshape(x_construction[9999,:],(28,28)), interpolation='nearest', cmap=cm.Greys)
ax4.set_title('Learned image')
plt.show()

z1 = z[:,0]
z2 = z[:,1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z1,z2,'ko')
plt.title("latent space")

#np.where((z1>3) & (z2<2) & (z2>0))
#select the points from the latent space
a_vec = [2,5,7,789,25,9993]
for i in range(len(a_vec)):
    ax.plot(z1[a_vec[i]],z2[a_vec[i]],'ro')  
    ax.annotate('z%d' %i, xy=(z1[a_vec[i]],z2[a_vec[i]]), 
                xytext=(z1[a_vec[i]],z2[a_vec[i]]),color = 'r',fontsize=15)


f, ((ax0, ax1, ax2, ax3, ax4,ax5)) = plt.subplots(1,6,  sharex='col', sharey='row',figsize=(12,2.5))
for i in range(len(a_vec)):
    eval('ax%d' %(i)).imshow(np.reshape(x_construction[a_vec[i],:],(28,28)), interpolation='nearest', cmap=cm.Greys)
    eval('ax%d' %(i)).set_title('z%d'%i)

plt.show()

