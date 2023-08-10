#! Setup:

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns#nicer plots?
sns.reset_orig()
import scipy.stats as sps
from scipy.stats import chi
import scipy as sp
from scipy import signal
from sklearn import linear_model
import pickle
import datetime
import time
import os
fig_size = [18,18]
plt.rcParams["figure.figsize"] = fig_size

# Tensorflow needs to be LAST import
import tensorflow as tf
print(tf.__version__)

#! Loading the data from VGG Conv2 activations
Data=np.load('/gpfs01/bethge/home/dklindt/David/publish/fig5/more_types/data_all.npz')
print(Data.keys())
Y=Data.f.responses
Y=Y.T# N x D
X=Data.f.images
X=np.transpose(X,axes=[0,3,1,2])# NCHW
loc_y=Data.f.loc_y
loc_x=Data.f.loc_x
feature_maps=Data.f.feature_maps

#! splits data into train,val,test and adds poisson-like noise
def splitting_data(X,Y,num_train,num_val,num_test=2**10,noise=True,mean_response=.1):
    
    N=Y.shape[0]#number of neurons
    
    X_train=X[:num_train,:,:,:]
    X_val=X[num_train:num_train+num_val,:,:,:]
    X_test=X[num_train+num_val:num_train+num_val+num_test,:,:,:]
    
    Y_train=np.zeros([N,num_train])
    Y_val=np.zeros([N,num_val])
    #Y_test=np.zeros([N,num_test])
    GT_test=np.zeros([N,num_test])
    
    for n in range(1000):
        tmp_mean = np.mean(Y[n,:])
        Y_train[n,:] = Y[n,:num_train] / tmp_mean * mean_response
        Y_val[n,:] = Y[n,num_train:num_train+num_val] / tmp_mean * mean_response
        GT_test[n,:] = Y[n,num_train+num_val:num_train+num_val+num_test] / tmp_mean * mean_response
    
    #Poisson-like noise
    if noise:
        Y_train += np.random.normal(0,np.sqrt(np.abs(Y_train)),Y_train.shape)
        Y_val +=  np.random.normal(0,np.sqrt(np.abs(Y_val)),Y_val.shape)
        #Y_test =  GT_test + np.random.normal(0,np.sqrt(np.abs(GT_test)),GT_test.shape)
    
    #Poisson noise
    #if noise:
    #    Y_train += np.random.poisson(Y_train)
    #    Y_val +=  np.random.poisson(Y_val)
    #    Y_test =  GT_test + np.random.poisson(GT_test)
    
    return Y_train,Y_val,X_train,X_val,X_test,GT_test#,Y_test

#! Spike triggered average - as initialization or else
def STA(X,Y,crop,s2=13):
    #In:
    #X - stimuli, stimulus_size x number_of_data
    #Y - responses, number_of_neurons x number_of_data
    #smooth - size of smoothing gaussian (=s2), if not provided, no smoothing
    #Out
    #sta - spike triggered average, number_of_neurons x stimulus_size
    s=np.sqrt(X.shape[0]).astype(int)
    d=X.shape[1]
    n=Y.shape[0]
    
    sta = ((X.dot(Y.T))/Y.shape[1]).T
    sta = sta.reshape([n,s*s])
    
    #Smoothing
    x = np.linspace(1, s2, s2)
    y = np.linspace(s2, 1, s2)
    xm, ym = np.meshgrid(x, y)

    centre = [s2/2+.5, s2/2+.5]
    ind_tmp = (np.abs(xm-centre[0]) < s2) & (np.abs(ym-centre[1]) < s2)
    rf_tmp = np.zeros((s2,s2))
    rf_tmp[ind_tmp] = np.sqrt( (centre[0] - xm[ind_tmp])**2 +
                      (centre[1] - ym[ind_tmp])**2 )
    rf_tmp[ind_tmp] = (sps.norm.pdf(rf_tmp[ind_tmp],0,s2**(1/4)))

    normal=rf_tmp
    #plt.imshow(normal)
    #plt.show()
    sta_smooth=np.zeros(sta.shape)
    sta_r1=np.zeros([n,s,s])
    for i in range(n):
        sta_smooth[i,:] = signal.convolve2d(((sta[i,:])**2).reshape([s,s]),
            normal,mode='same').reshape(s**2)
        #rank one approx
        U,S,V = np.linalg.svd(sta[i,:].reshape(s,s))
        S1 = np.zeros(S.shape)
        S1[0] = S[0]
        tmp1 = U.dot(np.diag(S1).dot(V))
        sta_r1[i,:,:] = signal.convolve2d((tmp1**2),
            normal,mode='same')
        
    
    
    #cropping
    ind = np.int((s-crop)/2)
    sta = sta.reshape([n,s,s])
    sta = sta[:,ind:s-ind,ind:s-ind]
    sta = sta.reshape([n,crop**2])
    sta_s = sta_smooth.reshape([n,s,s])
    sta_s = sta_s[:,ind:s-ind,ind:s-ind]
    sta_s = sta_s.reshape([n,crop**2])
    sta_r1 = sta_r1[:,ind:s-ind,ind:s-ind]
    sta_r1 = sta_r1.reshape([n,crop**2])
        
    return sta, sta_s#, sta_r1


#######rank one approx
'''
for i in range(3):
    tmp = tmp_sta[356+i,:].reshape(32,32).copy()
    plt.imshow(tmp)
    plt.show()
    u,s,v = np.linalg.svd(tmp)
    print(u.shape,s.shape,v.shape)
    s1 = np.zeros(s.shape)
    s1[0] = s[0]
    tmp1 = u.dot(np.diag(s1).dot(v))
    plt.imshow(tmp1)
    plt.show()
'''

#! Model tensorflow
from tensorflow.contrib import layers
class ModelGraph:
    def __init__(self, s, rT, rA, init_scaleK, init_scaleT, N, num_kern,
                 init_mask=np.array([]),
                 init_weights=np.array([]), init_kernel=np.array([])):
        #Inputs:
        #        s*s - size of the image
        #        s2*s2 - size of the kernel
        #        rM/W - regularization weight Mask / Weights
        #        N - number of neurons
        #        num_kern - number of kernels per conv layer
        
        self.graph = tf.Graph()#new tf graph
        with self.graph.as_default():#use it as default
            
            tf_seed=3973
            #input tensor of shape NCHW!
            self.X = tf.placeholder(tf.float32,shape=[None,3,s[0],s[0]])
            #output: N x None
            self.Y = tf.placeholder(tf.float32)             
            
            #WK Kernel - filter / tensor of shape H-W-InChannels-OutChannels
            
            #batch normalization settings
            self.istrain = tf.placeholder(tf.bool)
            
            bn_params = dict(center=True,
                             scale=False,
                             is_training=self.istrain,
                             variables_collections=['batch_norm_ema'])
            
            #Layer: Conv1
            self.conv1 = layers.convolution2d(
                inputs=self.X,
                data_format='NCHW',
                num_outputs=num_kern[0],
                kernel_size=s[1],
                stride=1,
                padding='VALID',
                activation_fn=tf.nn.relu,#None
                normalizer_fn=layers.batch_norm,
                normalizer_params=bn_params,
                weights_initializer=tf.random_normal_initializer(stddev=init_scaleK),
                scope='conv1')
            with tf.variable_scope('conv1', reuse=True):
                self.WK1 = tf.get_variable('weights')
                
            #Layer: Conv2
            self.conv2 = layers.convolution2d(
                inputs=self.conv1,
                data_format='NCHW',
                num_outputs=num_kern[1],
                kernel_size=s[1],
                stride=1,
                padding='VALID',
                activation_fn=tf.nn.relu,#None
                normalizer_fn=layers.batch_norm,
                normalizer_params=bn_params,
                weights_initializer=tf.random_normal_initializer(stddev=init_scaleK),
                scope='conv2')
            with tf.variable_scope('conv2', reuse=True):
                self.WK2 = tf.get_variable('weights')
                
            #Layer: Conv3
            self.conv3 = layers.convolution2d(
                inputs=self.conv2,
                data_format='NCHW',
                num_outputs=num_kern[2],
                kernel_size=s[1],
                stride=1,
                padding='VALID',
                activation_fn=tf.nn.relu,#None
                normalizer_fn=layers.batch_norm,
                normalizer_params=bn_params,
                weights_initializer=tf.random_normal_initializer(stddev=init_scaleK),
                scope='conv3')
            with tf.variable_scope('conv3', reuse=True):
                self.WK3 = tf.get_variable('weights')
            
            #batch_norm op
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            #WT read out tensor
            self.WT_init = tf.random_normal([num_kern[-1],s[2],s[2],N],init_scaleT[0],init_scaleT[1])
            self.WT = tf.Variable(self.WT_init,dtype=tf.float32,name='WT')
            
            #Predicted Output
            self.Y_ = tf.transpose(tf.einsum('dchw,chwn->dn',self.conv3,self.WT))#NxD
            
            #Regularization
            self.regT = tf.reduce_sum(tf.square(self.WT)) #L2 Loss on read-out tensor
            self.regA = tf.reduce_mean(tf.reduce_sum(tf.abs(self.Y_),0)) #mean L1 Loss on Activations
            
            #Define a loss function
            self.res = self.Y_-self.Y
            self.MSE = tf.reduce_sum(tf.reduce_mean(self.res * self.res,1))
            self.loss = self.MSE + rT*self.regT + rA*self.regA
            
            #Define a training graph
            self.step_size= tf.placeholder(tf.float32)
            self.training = tf.train.AdamOptimizer(self.step_size).minimize(self.loss)
            
            # Create a saver.
            self.saver = tf.train.Saver()

#! Training CNN-NL
def train(init_scaleK,init_scaleT,init_lr,num_kern,max_runs,rT,rA,s,N,
          X_train,X_val,X_test,Y_train,Y_val,GT_test,batch_size,verbose=False):
    
    #Storing:
    MSE_train = [] # MSE on train set, reps x runs/100
    tmp_MT = []#dummy for storing the above during repetition for Xruns
    MSE_val = []#MSE on validation set, reps x runs/100
    tmp_MV = []#dummy for storing the above during repetition for Xruns
    MSE_test = []#MSE on test set, reps x 1
    WK1 = []# Kernel - store best weights, reps x s2 x s2
    tmp_WK1 = []#dummy during run
    WK2 = []# Kernel - store best weights, reps x s2 x s2
    tmp_WK2 = []#dummy during run
    WK3 = []# Kernel - store best weights, reps x s2 x s2
    tmp_WK3 = []#dummy during run
    WT = []# Read Out Weights - store best weights, reps x s x s
    tmp_WT = []#dummy during run
    FEV = []#fraction of explained variance, reps x 1

    #calculate test variance
    gt_test_var = np.sum(np.var(GT_test,axis =1))#explainable output variance
    
    #initialize current attributes
    lr=init_lr
    # flags for early stopping
    stop_flag = 0
    sstop=0

    #Init model class
    model = ModelGraph(s,rT,rA,init_scaleK,init_scaleT,N,num_kern)

    #validation feed can be outside loop:
    #feed_val = {model.X:X_val,
    #            model.Y:Y_val,model.istrain:False}
    #feed_test = {model.X:X_test,model.Y:GT_test,model.istrain:False}
    
    #Split up because too large for GPU memory...
    num_test=np.int(GT_test.shape[1]/batch_size)
    feed_test=[]
    for i in range(num_test):
        feed_test.append( {model.X:X_test[i*batch_size:(i+1)*batch_size,:,:,:],
                        model.Y:GT_test[:,i*batch_size:(i+1)*batch_size],model.istrain:False})
    num_val=np.int(Y_val.shape[1]/batch_size)
    feed_val=[]
    for i in range(num_val):
        feed_val.append( {model.X:X_val[i*batch_size:(i+1)*batch_size,:,:,:],
                        model.Y:Y_val[:,i*batch_size:(i+1)*batch_size],model.istrain:False})
    
    ##Start a tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with model.graph.as_default():
        with tf.Session(config=config) as sess:
            
            tf_seed=3973
            sess.run(tf.global_variables_initializer())
            
            #plot initial weights
            if verbose:
                print('Before Training:')
                fig, ax = plt.subplots(1, 5, figsize=[18, 3])
                tmp_wt=model.WT.eval()[...,0]#first neuron as example
                for i in range(4):
                    ax[i].imshow(tmp_wt[i,:,:].reshape(s[2],s[2]),cmap='bwr',
                                 vmin=-np.max(abs(tmp_wt[i,:,:])),
                                 vmax=np.max(abs(tmp_wt[i,:,:])))
                test=np.zeros(num_test*batch_size)
                act_loss=np.zeros(num_test)
                for i in range(num_test):
                    [tmp_y,
                     act_loss[i]] = sess.run([model.Y_,model.regA],feed_test[i])
                    test[i*batch_size:(i+1)*batch_size] = tmp_y[0]
                ax[4].plot(GT_test[0], test, '.')
                xx = [-.4, .4]
                ax[4].plot(xx, xx)
                ax[4].axis('equal')
                ax[4].set_title('true vs predicted')
                plt.show()
                print('Loss: %s tensor'%(model.regT.eval()*rT),
                      'Loss: %s activation'%(np.mean(act_loss)*rA))

            #Batches - define list of starting-indices for individual batches in data set:
            #if there is less training data than batch size
            batch_size = np.min([batch_size,X_train.shape[0]])
            batch_ind = np.arange(0,X_train.shape[0],batch_size)
            #number of selected batch
            batch = 0

            for j in range(1,max_runs):

                #when there is no further complete batch
                if batch==len(batch_ind):
                    #shuffle data and start again:
                    ind = np.random.permutation(X_train.shape[0])
                    X_train = X_train[ind,:,:,:]
                    Y_train = Y_train[:,ind]
                    batch = 0

                #take a batch
                X_batch = X_train[batch_ind[batch]:batch_ind[batch]+batch_size,:,:,:]
                Y_batch = Y_train[:,batch_ind[batch]:batch_ind[batch]+batch_size]
                batch +=1
                
                #Training feed:
                feed_dict ={model.step_size:lr,model.X:X_batch,
                            model.Y:Y_batch,model.istrain:True}

                # Training with current batch:
                sess.run([model.training, model.update_ops],feed_dict)
                
                #Early Stopping - check if MSE doesn't increase
                if j%100==0:

                    model.saver.save(sess, 'McInt_bn_checkpoint', global_step=int(j/100))

                    # Store MSE on train:
                    
                    tmp_MT.append(model.MSE.eval(feed_dict))

                    #check MSE on validation set and store the parameters
                    #tmp_MV.append(model.MSE.eval(feed_val))
                    val=np.zeros(num_val)
                    for i in range(num_val):
                        val[i]=sess.run(model.MSE,feed_val[i])
                    tmp_MV.append(np.mean(val))
                    tmp_WK1.append(model.WK1.eval())
                    tmp_WK2.append(model.WK2.eval())
                    tmp_WK3.append(model.WK3.eval())
                    tmp_WT.append(model.WT.eval())

                    #Best run
                    if len(tmp_MV)>5:#burn in 
                        tmp_min_ind = np.argmin(tmp_MV[5:])+5
                    else:
                        tmp_min_ind = len(tmp_MV)-1

                    ##Analytics - Display progress
                    if verbose:
                        #to calculate FEV
                        test=np.zeros(num_test)
                        for i in range(num_test):
                            test[i]=sess.run(model.MSE,feed_test[i])
                        MSE_gt = np.mean(test)
                        print('Runs: %s; MSE - train: %s, val: %s; lr = %s'%
                              (j,tmp_MT[-1],tmp_MV[-1],lr))
                        print('latest: ',tmp_MV[-1])
                        print('FEV = ',(1 - (MSE_gt/gt_test_var)))
                        print('best: ',tmp_min_ind,tmp_MV[tmp_min_ind])
                        
                        fig, ax = plt.subplots(1, 6, figsize=[18, 3])
                        tmp_wt=model.WT.eval()[...,0]#first neuron as example
                        for i in range(4):
                            ax[i].imshow(tmp_wt[i,:,:].reshape(s[2],s[2]),cmap='bwr',
                                         vmin=-np.max(abs(tmp_wt[i,:,:])),
                                         vmax=np.max(abs(tmp_wt[i,:,:])))
                        test=np.zeros(num_test*batch_size)
                        act_loss=np.zeros(num_test)
                        for i in range(num_test):
                            [tmp_y,
                             act_loss[i]] = sess.run([model.Y_,model.regA],feed_test[i])
                            test[i*batch_size:(i+1)*batch_size] = tmp_y[0]
                        ax[4].plot(GT_test[0], test, '.')
                        xx = [-.4, .4]
                        ax[4].plot(xx, xx)
                        ax[4].axis('equal')
                        ax[4].set_title('true vs predicted')
                        ax[5].plot(tmp_MV)
                        ax[5].plot(tmp_MT)
                        ax[5].set_ylim([min(tmp_MT),2*np.median(tmp_MV)-min(tmp_MT)])
                        ax[5].legend(['MSE Val','MSE Train'])
                        plt.show()
                        print('Loss: %s tensor'%(model.regT.eval()*rT),
                              'Loss: %s activation'%(np.mean(act_loss)*rA))
                        

                    ##Early Stopping - if latest validation MSE is not minimum
                    if tmp_min_ind != len(tmp_MV)-1:
                        stop_flag +=1
                        if stop_flag>=8:
                            lr *= .1
                            #set back to previous best?
                            #model.saver.restore(sess, 'bn_checkpoint-%s'%(tmp_min_ind+1))
                            #print('back to MSE-val = ',model.MSE_test.eval(feed_val))
                            stop_flag = 0
                            sstop +=1
                            if sstop==3:#lower the lr x times
                                break
                                
                    else:#if latest value is best, reset
                        stop_flag = 0

            #Best run
            tmp_min_ind = np.argmin(tmp_MV)

            #Store MSEs
            MSE_train = tmp_MT#list
            MSE_val = tmp_MV#list
            
            #Store best weights (i.e. lowest validation MSE)
            WK1 = tmp_WK1[tmp_min_ind]#s2 x s2
            WK2 = tmp_WK2[tmp_min_ind]#s2 x s2
            WK3 = tmp_WK3[tmp_min_ind]#s2 x s2
            WT = tmp_WT[tmp_min_ind]#

            #Assign the best weights to model graph 
            model.saver.restore(sess, './McInt_bn_checkpoint-%s'%(tmp_min_ind+1))
            
            #clean checkpoints
            files = os.listdir()
            for file in files:
                if file.startswith("McInt_bn_checkpoint"):
                    os.remove(file)
                    
            # Test MSE prediction
            #MSE_test = sess.run(model.MSE,feed_test)#1 x 1
            test=np.zeros(num_test)
            #test_L=np.zeros(num_test)
            for i in range(num_test):
                test[i]=sess.run(model.MSE,feed_test[i])
            #    test_L[i]=sess.run(model.poisson,feed_test[i])
            MSE_test = np.mean(test)
            #Loss_test = np.mean(Loss)
            
            #FEV - fraction of explainable variance
            FEV = 1 - (MSE_test/gt_test_var)#1 x 1

            #Predicted outputs
            #Y_ = sess.run(model.Y_,feed_test)
            test=np.zeros([N,num_test*batch_size])
            for i in range(num_test):
                test[:,i*batch_size:(i+1)*batch_size]=sess.run(model.Y_,feed_test[i])
            Y_ = test
            
            #FEV per cell
            fev_cell=1-(np.mean((Y_-GT_test)**2,1)/np.var(GT_test,1))

            #Output
            log=('Stop at run %s; MSE on validation set: %s'% (j,MSE_val[tmp_min_ind]),
                  'MSE on test set: %s; FEV: %s' % (MSE_test, FEV))
            print(log)

    return (WK1,WK2,WK3,WT,MSE_train,MSE_val,MSE_test,FEV,fev_cell,Y_,log)

#good reg values found with previous exploration
rA = [.1,.3]
rT = [1,3]

### grid search
reps = 5

Neurons = [4*64,8*64,16*64]

start=time.time()
#Seeds
np_seed=1234
np.random.seed(np_seed)
tf_seed=1234

#Data, keep test set for later
N=Neurons[-1]
num_test = 2**10
ind = np.random.choice(X.shape[0],X.shape[0],replace=False)
X_TEST = X[ind[:num_test],...]
GT_TEST = ((Y[:,ind[:num_test]].T / np.mean(Y,1)).T * .1)[:N,:]#mean response=.1
X_try = X[ind[num_test:],...]
Y_try = Y[:N,ind[num_test:]]

#Data
D=2**12
split_data=True#set true for large nets to fit on GPU memory
batch_size = 64
D=min(D,Y_try.shape[1])#number of training+val images
num_val = np.min([D//8,num_test])#limit validation set to test set size
num_train = D-num_val

Y_train,Y_val,X_train,X_val,X_test,GT_test = splitting_data(X=X_try,Y=Y_try,
   num_train=num_train,num_test=num_test,num_val=num_val,noise=True,mean_response=.1)

print('Images for Training: %s, Validation: %s, Testing %s'%(
      Y_train.shape[1],Y_val.shape[1],GT_test.shape[1]))

#Ground truth locations
GT_mask = np.hstack([loc_y.reshape([-1,1]),loc_x.reshape([-1,1])])#[ind,:]

#Parameters:
s1=44#width=heigth of image
s2=5
s3=32
n_B = [32,64,0]#number of bases to learn
init_scaleK = .01#Kernel
init_scaleT = [0,.01]#Read out Weights
lr= .001 # initial learning rate
#loop parameters
max_runs = 15000 # training steps
batch_size = 64
    
#Clean previous checkpoints
files = os.listdir()
for file in files:
    if file.startswith("McInt_bn_checkpoint"):
        os.remove(file)


print('it took %s s to preprocess data'%(time.time()-start))

Val_c4 = np.zeros([reps,4])#rep,reg (mean over neurons)
Fev_c4 = np.zeros([reps,4,Neurons[0]])
Val_c8 = np.zeros([reps,4])#rep,reg (mean over neurons)
Fev_c8 = np.zeros([reps,4,Neurons[1]])
Val_c16 = np.zeros([reps,4])#rep,reg (mean over neurons)
Fev_c16 = np.zeros([reps,4,Neurons[2]])

for rep in range(reps):
    with open("mcint_log.txt", "a") as file:
        print('repetition ',rep, file=file)
    for n in range(len(Neurons)):
        n_B[2] = Neurons[n]//64
        with open("mcint_log.txt", "a") as file:
            print('neurons %s, kernels/types %s'%(Neurons[n],n_B[2]),file=file)
        for regA in range(len(rA)):
            with open("mcint_log.txt", "a") as file:
                print('regularization on activations = ',rA[regA],file=file)
            for regT in range(len(rT)):
                with open("mcint_log.txt", "a") as file:
                    print('regularization on tensor = ',rT[regT],file=file)
                    
                (WK1,WK2,WK3,WT_tmp,MSE_train,MSE_val,MSE_test,
                FEV,fev_c,Y_tmp,log) = train(init_scaleK,
                           init_scaleT,lr,n_B,
                           max_runs,rT[regT],rA[regA],[s1,s2,s3],Neurons[n],X_train,X_val,X_TEST,
                           Y_train[:Neurons[n],:],Y_val[:Neurons[n],:],GT_TEST[:Neurons[n],:],
                            batch_size,verbose=False)
                
                #flatten the grid search over regs
                reg=regA*2+regT
                
                if n==0:
                    Val_c4[rep,reg] = np.min(MSE_val)
                    Fev_c4[rep,reg,:] = fev_c

                if n==1:
                    Val_c8[rep,reg] = np.min(MSE_val)
                    Fev_c8[rep,reg,:] = fev_c

                if n==2:
                    Val_c16[rep,reg] = np.min(MSE_val)
                    Fev_c16[rep,reg,:] = fev_c

                #saving text output to log file
                with open("mcint_log.txt", "a") as file:
                    print(log, file=file)
                np.savez('mcint_4-8-16_types',
                         Val_c4=Val_c4,
                         Fev_c4=Fev_c4,
                         Val_c8=Val_c8,
                         Fev_c8=Fev_c8,
                         Val_c16=Val_c16,
                         Fev_c16=Fev_c16)

