import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#https://www.tensorflow.org/get_started/mnist/beginners

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(help(mnist))

print(help(mnist.train))

print(mnist.train.images.shape)
print(mnist.train.labels.shape)


def get_previous_features(i_layer):
    convx_dims = i_layer.get_shape().as_list()
    output_features = 1
    for dim in range(1,len(convx_dims)):
        output_features=output_features*convx_dims[dim]
    return output_features



def conv(input_matrix,filter_size=3,layer_depth=8,
              strides=[1,1,1,1],padding='SAME',
              is_training=True,name_scope="lx",
              stddev_n = 0.05,
             max_bool=False,max_kernel=[1,2,2,1],max_strides=[1,1,1,1], max_padding='SAME',
             drop_out_bool=False,drop_out_ph=None,drop_out_v=None,decay=0.5
             ):
    with tf.name_scope(name_scope):
        ims = input_matrix.get_shape().as_list()
        input_depth=ims[len(ims)-1]
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,input_depth,layer_depth], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[layer_depth]),name='b')
        c = tf.add(tf.nn.conv2d(input_matrix, W, strides=strides, padding=padding),b,name='conv')
        n = tf.contrib.layers.batch_norm(c, center=True, scale=True, is_training=is_training,decay=decay)
        a = tf.nn.relu(n,name="activation")
        if max_bool==True:
            out = tf.nn.max_pool(a, ksize=max_kernel,strides=max_strides, padding=max_padding,name='max')
        else:
            out = a
        if drop_out_bool==True:
            out_  = tf.nn.dropout(out, drop_out_ph)
        else:
            out_ = out
        return out_


def fc(input_matrix,n=22,norm=False,prev_conv=False,
       stddev_n = 0.05,is_training=True,
       name_scope='FC',drop_out_bool=False,drop_out_ph=None,drop_out_v=None,decay=0.5):
    with tf.name_scope(name_scope):
        cvpfx = get_previous_features(input_matrix)
        if prev_conv==True:
            im = tf.reshape(input_matrix, [-1, cvpfx])
        else:
            im = input_matrix
        W = tf.Variable(tf.truncated_normal([cvpfx, n], stddev=stddev_n),name='W')
        b = tf.Variable(tf.constant(stddev_n, shape=[n]),name='b') 
        fc = tf.add(tf.matmul(im, W),b,name="FC")
        if name_scope=="FCL":
            out_ = fc
        else:
            if norm==True:
                n = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=is_training,decay=decay)
                out = tf.nn.relu(n,name="activation")
            else:
                out = tf.nn.relu(fc,name="activation")
            if drop_out_bool==True:
                out_  = tf.nn.dropout(out, drop_out_ph)
            else:
                out_ = out
        return out_

def train(ix,iy,iters=10,lr=0.001,save_model=True,save_name=None,restore_model=False,restore_name=None,v=False):
    "v: for verbosity"
    tf.reset_default_graph()
    class_output = iy.shape[1]
    d0 = ix.shape[0]
    x_shape=[None]
    for _ in range(1,len(ix.shape)):
        x_shape.append(ix.shape[_])
    xi = tf.placeholder(tf.float32, shape=x_shape,name='x')
    y_ = tf.placeholder(tf.float32, shape=[None,class_output],name='y')
    train_bool=tf.placeholder(bool,name='train_test')
    learning_rate = tf.placeholder(tf.float32)
    
    #Define the model here--DOWN
    CV1 = conv(xi,filter_size=3,layer_depth=2,name_scope="CL1",is_training=train_bool)
    CV2 = conv(CV1,filter_size=3,layer_depth=2,name_scope="CL2",is_training=train_bool)
    prediction = fc(CV2,n=class_output,name_scope="FCL",prev_conv=True)
    #Define the model here--UP
    
    y_CNN = tf.nn.softmax(prediction,name='Softmax')
    class_pred = tf.argmax(y_CNN,1,name='ClassPred')
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
    
    #The following three lines are required to make "is_training" work for normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as s:
        if restore_model==True:
            if restore_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(s,restore_dir)
        else:
            s.run(init_op)
        fd={xi:ix,y_:iy,learning_rate:lr,train_bool:True}
        for _ in range(0,iters):
            _t,l = s.run([train_step,loss],feed_dict=fd)
            if v==True:
                print("Iter:",_,"Loss:",l)
            
        if save_model==True:
            if save_name==None:
                print("No model specified, model not being saved")
                return
            else:
                save_path = saver.save(s, save_name)
                print("Model saved in file: %s" % save_name)

base_model = 'MNIST/mnist_01_.ckpt'

mtx = mnist.train.images
mtx_s = mtx.shape
x_train = np.reshape(mtx,[mtx_s[0],int(mtx_s[1]**0.5),int(mtx_s[1]**0.5),1])
print("Train shape",mtx_s)
print(x_train.shape)

limit = 16
shift = 0
y_train = mnist.train.labels

xt = x_train[shift:shift+limit,:]
yt = y_train[shift:shift+limit,:]

train(ix=xt,iy=yt,iters=10,lr=0.001,save_model=True,save_name=base_model,restore_model=False,restore_name=None)

def restore_see_layer(ix,model_name=None,var_name=None):
    with tf.Session('', tf.Graph()) as s:
        with s.graph.as_default():
            if (model_name!=None) and var_name!=None:
                saver = tf.train.import_meta_graph(model_name+".meta")
                saver.restore(s,model_name)
                fd={'x:0':ix,'train_test:0':False}
                var_name=var_name+":0"
                result = s.run(var_name,feed_dict=fd)
    return result

test_img = xt[0:1,:]

output_cl1 = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL1/conv')
print(output_cl1.shape)

output_cl2 = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL2/conv')
print(output_cl2.shape)

def see_output(iNp,depth_filter_to_see=0,cmap="gray",figsize=(4,4)):
    img_x = iNp[0,:,:,depth_filter_to_see]
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_x,cmap=cmap)
    plt.show()

see_output(output_cl1,1)
see_output(output_cl2,1)

train(ix=xt,iy=yt,iters=1000,lr=0.001,save_model=True,save_name=base_model,restore_model=False,restore_name=None,v=True)

output_cl1 = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL1/conv')
print(output_cl1.shape)

output_cl2 = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL2/conv')
print(output_cl2.shape)

see_output(output_cl1,1)
see_output(output_cl2,1)

def see_filter(iNp,depth_filter_to_see=0,input_depth_filter_to_see=0,cmap="gray",figsize=(4,4)):
    img_x = iNp[:,:,input_depth_filter_to_see,depth_filter_to_see]
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_x,cmap=cmap)
    plt.show()


output_cl1_filter = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL1/W')
print(output_cl1_filter.shape)

output_cl2_filter = restore_see_layer(ix=test_img,model_name=base_model,var_name='CL2/W')
print(output_cl2_filter.shape)

see_filter(output_cl1_filter,1)
see_filter(output_cl2_filter,1)

def train_max(ix,iy,iters=10,lr=0.001,save_model=True,save_name=None,restore_model=False,restore_name=None,v=False):
    "v: for verbosity"
    tf.reset_default_graph()
    class_output = iy.shape[1]
    d0 = ix.shape[0]
    x_shape=[None]
    for _ in range(1,len(ix.shape)):
        x_shape.append(ix.shape[_])
    xi = tf.placeholder(tf.float32, shape=x_shape,name='x')
    y_ = tf.placeholder(tf.float32, shape=[None,class_output],name='y')
    train_bool=tf.placeholder(bool,name='train_test')
    learning_rate = tf.placeholder(tf.float32)
    
    CV1 = conv(xi,filter_size=3,layer_depth=2,name_scope="CL1",is_training=train_bool,max_bool=True)
    CV2 = conv(CV1,filter_size=3,layer_depth=2,name_scope="CL2",is_training=train_bool,max_bool=True)
    prediction = fc(CV2,n=class_output,name_scope="FCL",prev_conv=True)
    
    y_CNN = tf.nn.softmax(prediction,name='Softmax')
    class_pred = tf.argmax(y_CNN,1,name='ClassPred')
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as s:
        if restore_model==True:
            if restore_name==None:
                print("No model file specified")
                return
            else:
                saver.restore(s,restore_dir)
        else:
            s.run(init_op)
        fd={xi:ix,y_:iy,learning_rate:lr,train_bool:True}
        for _ in range(0,iters):
            _t,l = s.run([train_step,loss],feed_dict=fd)
            if v==True:
                print("Iter:",_,"Loss:",l)
            
        if save_model==True:
            if save_name==None:
                print("No model specified, model not being saved")
                return
            else:
                save_path = saver.save(s, save_name)
                print("Model saved in file: %s" % save_name)

model_max= 'MNIST/model_max2l_.ckpt'
train_max(ix=xt,iy=yt,iters=10,lr=0.001,save_model=True,save_name=model_max,restore_model=False,restore_name=None)

output_cl1 = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL1/max')
print(output_cl1.shape)

output_cl2 = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL2/max')
print(output_cl2.shape)

see_output(output_cl1,1)
see_output(output_cl2,1)

output_cl1_filter = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL1/W')
print(output_cl1_filter.shape)

output_cl2_filter = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL2/W')
print(output_cl2_filter.shape)

see_filter(output_cl1_filter,1)
see_filter(output_cl2_filter,1)

train_max(ix=xt,iy=yt,iters=1000,lr=0.001,save_model=True,save_name=model_max,restore_model=False,restore_name=None,v=True)

output_cl1 = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL1/max')
print(output_cl1.shape)

output_cl2 = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL2/max')
print(output_cl2.shape)

see_output(output_cl1,1)
see_output(output_cl2,1)

output_cl1_filter = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL1/W')
print(output_cl1_filter.shape)

output_cl2_filter = restore_see_layer(ix=test_img,model_name=model_max,var_name='CL2/W')
print(output_cl2_filter.shape)

see_filter(output_cl1_filter,1)
see_filter(output_cl2_filter,1)



