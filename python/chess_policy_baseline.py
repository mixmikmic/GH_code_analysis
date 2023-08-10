get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from cchess import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random 
import time
from utils import Dataset,ProgressBar
from tflearn.data_flow import DataFlow,DataFlowStatus,FeedDictFlow
from tflearn.data_utils import Preloader,ImagePreloader
import scipy
import pandas as pd
import xmltodict
from game_convert import convert_game
import tflearn

GPU_CORE = 0
BATCH_SIZE = 256
BEGINING_LR = 0.01
#TESTIMG_WIDTH = 500
model_name = '11_4_resnet'
data_dir = 'data/imsa-cbf/'

class ElePreloader(object):
    def __init__(self,datafile,batch_size=64):
        self.batch_size=batch_size
        content = pd.read_csv(datafile,header=None,index_col=None)
        self.filelist = [i[0] for i in content.get_values()]
        self.pos = 0
        self.feature_list = {"red":['A', 'B', 'C', 'K', 'N', 'P', 'R']
                             ,"black":['a', 'b', 'c', 'k', 'n', 'p', 'r']}
        self.batch_size = batch_size
        self.batch_iter = self.__iter()
        assert(len(self.filelist) > batch_size)
        self.game_iterlist = [None for i in self.filelist]
    
    def __iter(self):
        retx1,rety1,retx2,rety2 = [],[],[],[]
        filelist = []
        while True:
            for i in range(self.batch_size):
                if self.game_iterlist[i] == None:
                    if len(filelist) == 0:
                        filelist = copy.copy(self.filelist)
                        random.shuffle(filelist)
                    self.game_iterlist[i] = convert_game(filelist.pop(),feature_list=self.feature_list)
                game_iter = self.game_iterlist[i]
                
                try:
                    x1,y1,x2,y2 = game_iter.__next__()
                    x1 = np.transpose(x1,[1,2,0])
                    x2 = np.transpose(x2,[1,2,0])
                    x1 = np.expand_dims(x1,axis=0)
                    x2 = np.expand_dims(x2,axis=0)
                    retx1.append(x1)
                    rety1.append(y1)
                    retx2.append(x2)
                    rety2.append(y2)
                    if len(retx1) >= self.batch_size:
                        yield (np.concatenate(retx1,axis=0),np.asarray(rety1)
                               ,np.concatenate(retx2,axis=0),np.asarray(rety2))
                        retx1,rety1,retx2,rety2 = [],[],[],[]
                except :
                    self.game_iterlist[i] = None

    def __getitem__(self, id):
        
        x1,y1,x2,y2 = self.batch_iter.__next__()
        return x1,y1,x2,y2
        
    def __len__(self):
        return 10000

trainset = ElePreloader(datafile='data/train_list.csv',batch_size=BATCH_SIZE)
with tf.device("/gpu:{}".format(GPU_CORE)):
    coord = tf.train.Coordinator()
    trainflow = FeedDictFlow({
            'data':trainset,
        },coord,batch_size=BATCH_SIZE,shuffle=True,continuous=True,num_threads=1)
trainflow.start()

testset = ElePreloader(datafile='data/test_list.csv',batch_size=BATCH_SIZE)
with tf.device("/gpu:{}".format(GPU_CORE)):
    coord = tf.train.Coordinator()
    testflow = FeedDictFlow({
            'data':testset,
        },coord,batch_size=BATCH_SIZE,shuffle=True,continuous=True,num_threads=1)
testflow.start()

sample_x1,sample_y1,sample_x2,sample_y2 = trainflow.next()['data']

sample_x1,sample_y1,sample_x2,sample_y2 = testflow.next()['data']

sample_x1.shape,sample_y1.shape,sample_x2.shape,sample_y2.shape

np.sum(sample_x2[0],axis=-1)

np.sum(sample_x1[0],axis=-1).shape

def res_block(inputx,name,training,block_num=2,filters=256,kernel_size=(3,3)):
    net = inputx
    for i in range(block_num):
        net = tf.layers.conv2d(net,filters=filters,kernel_size=kernel_size,activation=None,name="{}_res_conv{}".format(name,i),padding='same')
        net = tf.layers.batch_normalization(net,training=training,name="{}_res_bn{}".format(name,i))
        if i == block_num - 1:
            net = net + inputx #= tf.concat((inputx,net),axis=-1)
        net = tf.nn.elu(net,name="{}_res_elu{}".format(name,i))
    return net

def conv_block(inputx,name,training,block_num=1,filters=2,kernel_size=(1,1)):
    net = inputx
    for i in range(block_num):
        net = tf.layers.conv2d(net,filters=filters,kernel_size=kernel_size,activation=None,name="{}_convblock_conv{}".format(name,i),padding='same')
        net = tf.layers.batch_normalization(net,training=training,name="{}_convblock_bn{}".format(name,i))
        net = tf.nn.elu(net,name="{}_convblock_elu{}".format(name,i))
    # net [None,10,9,2]
    netshape = net.get_shape().as_list()
    print("inside conv block {}".format(str(netshape)))
    net = tf.reshape(net,shape=(-1,netshape[1] * netshape[2] * netshape[3]))
    net = tf.layers.dense(net,10 * 9,name="{}_dense".format(name))
    net = tf.nn.elu(net,name="{}_elu".format(name))
    return net

def res_net_board(inputx,name,training,filters=256):
    net = inputx
    net = tf.layers.conv2d(net,filters=filters,kernel_size=(3,3),activation=None,name="{}_res_convb".format(name),padding='same')
    net = tf.layers.batch_normalization(net,training=training,name="{}_res_bnb".format(name))
    net = tf.nn.elu(net,name="{}_res_elub".format(name))
    for i in range(NUM_RES_LAYERS):
        net = res_block(net,name="{}_layer_{}".format(name,i + 1),training=training)
        print(net.get_shape().as_list())
    print("inside res net {}".format(str(net.get_shape().as_list())))
    net_unsoftmax = conv_block(net,name="{}_conv".format(name),training=training)
    return net_unsoftmax

def get_scatter(name):
    with tf.variable_scope("Test"):
        ph = tf.placeholder(tf.float32,name=name)
        op = tf.summary.scalar(name,ph)
    return ph,op

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

NUM_RES_LAYERS = 10

with tf.device("/gpu:{}".format(GPU_CORE)):
    X1 = tf.placeholder(tf.float32,[None,10,9,14])
    y1 = tf.placeholder(tf.float32,[None,10,9])
    X2 = tf.placeholder(tf.float32,[None,10,9,15])
    y2 = tf.placeholder(tf.float32,[None,10,9])
    
    training = tf.placeholder(tf.bool,name='training_mode')
    learning_rate = tf.placeholder(tf.float32)
    global_step = tf.train.get_or_create_global_step()
    
    net_unsoftmax1 = res_net_board(X1,"selectnet",training=training)
    net_unsoftmax2 = res_net_board(X2,"movenet",training=training)
    
    target1 = tf.reshape(y1,(-1,10 * 9))
    target2 = tf.reshape(y2,(-1,10 * 9))
    with tf.variable_scope("Loss"):
        loss_select = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target1,logits=net_unsoftmax1))
        loss_move = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target2,logits=net_unsoftmax2))
        loss = loss_select + loss_move
        
        loss_select_summary = tf.summary.scalar("loss_select",loss_select)
        loss_move_summary = tf.summary.scalar("loss_move",loss_move)
        loss_summary = tf.summary.scalar("total_loss",loss)
    net_softmax1 = tf.nn.softmax(net_unsoftmax1)
    net_softmax2 = tf.nn.softmax(net_unsoftmax2)
    
    correct_prediction1 = tf.equal(tf.argmax(target1,1), tf.argmax(net_softmax1,1))
    correct_prediction2 = tf.equal(tf.argmax(target2,1), tf.argmax(net_softmax2,1))
    
    with tf.variable_scope("Accuracy"):
        accuracy_select = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        accuracy_move = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        accuracy_total = accuracy_select * accuracy_move
        
        acc_select_summary = tf.summary.scalar("accuracy_select",accuracy_select)
        acc_move_summary = tf.summary.scalar("accuracy_move",accuracy_move)
        acc_summary = tf.summary.scalar("acc_summary",accuracy_total)
        
    summary_op = tf.summary.merge([loss_select_summary,loss_move_summary,loss_summary
                                  ,acc_select_summary,acc_move_summary,acc_summary])
    
    test_select,test_select_summary = get_scatter("test_select_loss")
    test_move,test_move_summary = get_scatter("test_move_loss")
    test_total,test_total_summary = get_scatter("test_total_loss")
    test_selectacc,test_selectacc_summary = get_scatter("test_select_acc")
    test_moveacc,test_moveacc_summary = get_scatter("test_move_acc")
    test_totalacc,test_totalacc_summary = get_scatter("test_total_acc")
    
    test_summary_op = tf.summary.merge([test_select_summary,test_move_summary,test_total_summary
                                       ,test_selectacc_summary,test_moveacc_summary,test_totalacc_summary])
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        train_op = optimizer.minimize(loss,global_step=global_step)

    train_summary_writer = tf.summary.FileWriter("./log/{}_train".format(model_name), sess.graph)

sess.run(tf.global_variables_initializer())
tf.train.global_step(sess, global_step)

get_ipython().run_line_magic('pinfo2', 'tflearn.layers.residual_block')

import os
if not os.path.exists("models/{}".format(model_name)):
    os.mkdir("models/{}".format(model_name))

N_BATCH = 10000
N_BATCH_TEST = 300

restore = True
N_EPOCH = 100
DECAY_EPOCH = 10

class ExpVal:
    def __init__(self,exp_a=0.97):
        self.val = None
        self.exp_a = exp_a
    def update(self,newval):
        if self.val == None:
            self.val = newval
        else:
            self.val = self.exp_a * self.val + (1 - self.exp_a) * newval
    def getval(self):
        return round(self.val,2)
    
expacc = ExpVal()
expacc_select = ExpVal()
expacc_move = ExpVal()
exploss = ExpVal()


begining_learning_rate = 1e-1

pred_image = None
if restore == False:
    train_epoch = 1
    train_batch = 0
for one_epoch in range(train_epoch,N_EPOCH):
    train_epoch = one_epoch
    pb = ProgressBar(worksum=N_BATCH * BATCH_SIZE,info=" epoch {} batch {}".format(train_epoch,train_batch))
    pb.startjob()
    
    for one_batch in range(N_BATCH):
        if restore == True and one_batch < train_batch:
            pb.auto_display = False
            pb.complete(BATCH_SIZE)
            pb.auto_display = True
            continue
        else:
            restore = False
        train_batch = one_batch
        
        batch_x1,batch_y1,batch_x2,batch_y2 = trainflow.next()['data']
        # learning rate decay strategy
        batch_lr = begining_learning_rate * 10 ** -(one_epoch // DECAY_EPOCH)
        
        _,step_loss,step_acc_move,step_acc_select,step_acc_total,step_value,step_summary = sess.run(
            [train_op,loss,accuracy_move,accuracy_select,accuracy_total,global_step,summary_op],feed_dict={
                X1:batch_x1,y1:batch_y1,X2:batch_x2,y2:batch_y2,learning_rate:batch_lr,training:True
            })
        train_summary_writer.add_summary(step_summary,step_value)
        step_acc_move *= 100
        step_acc_select *= 100
        step_acc_total *= 100
        expacc.update(step_acc_total)
        expacc_select.update(step_acc_select)
        expacc_move.update(step_acc_move)
        exploss.update(step_loss)

       
        pb.info = "EPOCH {} STEP {} LR {} ACC {} SELACC{} MOVACC{} LOSS {} ".format(
            one_epoch,one_batch,batch_lr,expacc.getval()
            ,expacc_select.getval(),expacc_move.getval(),exploss.getval())
        
        pb.complete(BATCH_SIZE)
    print()
    accs = []
    accselects = []
    accmoves = []
    losses = []
    lossselects = []
    lossmoves = []
    pb = ProgressBar(worksum=N_BATCH_TEST * BATCH_SIZE,info="validating epoch {} batch {}".format(train_epoch,train_batch))
    pb.startjob()
    for one_batch in range(N_BATCH_TEST):
        batch_x1,batch_y1,batch_x2,batch_y2 = testflow.next()['data']
        step_loss_move,step_loss_select,step_loss,step_accuracy_move,step_accuracy_select,step_accuracy_total = sess.run(
            [loss_move,loss_select,loss,accuracy_move,accuracy_select,accuracy_total],feed_dict={
                X1:batch_x1,y1:batch_y1,X2:batch_x2,y2:batch_y2,training:False
            })
        accs.append(step_accuracy_total)
        accselects.append(step_accuracy_select)
        accmoves.append(step_accuracy_move)
        losses.append(step_loss)
        lossselects.append(step_loss_select)
        lossmoves.append(step_loss_move)
        
        pb.complete(BATCH_SIZE)
    print("TEST ACC {} SELACC {} MOVACC {} LOSS {}".format(np.average(accs),np.average(accselects)
                                                           ,np.average(accmoves),np.average(losses)))
    #test_select_summary,test_move_summary,test_total_summary
    #                                   ,test_selectacc_summary,test_moveacc_summary,test_totalacc_summary
    test_to_add_to_log = sess.run(test_summary_op,feed_dict={
        test_select:np.average(lossselects),test_move:np.average(lossmoves),test_total:np.average(losses)
        ,test_selectacc:np.average(accselects),test_moveacc:np.average(accmoves),test_totalacc:np.average(accs)
    })
    train_summary_writer.add_summary(test_to_add_to_log,step_value)
    print()
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.save(sess,"models/{}/model_{}".format(model_name,one_epoch))



