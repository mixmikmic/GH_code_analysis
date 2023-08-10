import tensorflow as tf
from keras.datasets import mnist
from keras import metrics
from config import myNet
from conv_net import utils
import numpy as np
from conv_net.var_autoencoder import variational_encoder
from plot import Plot
from conv_net import utils

from plot import Plot
import pandas as pd


from data_transformation.preprocessing import Preprocessing
from config import pathDict
from data_transformation.data_io import getH5File


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

out_img_shape = [64 ,64 ,3]
def run_preprocessor(sess, dataIN, preprocess_graph, is_training):
    out_shape = [dataIN.shape[0]] + out_img_shape
    pp_imgs = np.ndarray(shape=(out_shape), dtype='float32')
    for img_no in np.arange(dataIN.shape[0]):
        feed_dict = {
            preprocess_graph['imageIN']: dataIN[img_no, :],
            preprocess_graph['is_training']: is_training
        }
        pp_imgs[img_no, :] = sess.run(
                preprocess_graph['imageOUT'],
                feed_dict=feed_dict
        )

    return pp_imgs


def load_batch_data(image_type, which_data='cvalid'):
    if image_type not in ['bing_aerial', 'google_aerial', 'assessor', 'google_streetside', 'bing_streetside',
                          'google_overlayed', 'assessor_code']:
        raise ValueError('Can not identify the image type %s, Please provide a valid one' % (str(image_type)))

    data_path = pathDict['%s_batch_path' % (str(image_type))]
    batch_file_name = '%s' % (which_data)

    # LOAD THE TRAINING DATA FROM DISK
    dataX, dataY = getH5File(data_path, batch_file_name)

    return dataX, dataY



def RUN(epochs, batch_size):
    tf.reset_default_graph()

    preprocess_graph = Preprocessing(inp_img_shape=[224, 400 ,3],
                                     crop_shape=[128 ,128 ,3],
                                     out_img_shape=[64 ,64 ,3]).preprocessImageGraph()
    computation_graph = variational_encoder()
    test_loss = []
    training_loss = []
    test_accuracy = []
    learning_rate = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cvalidX, cvalidY = load_batch_data(image_type='assessor_code', which_data='cvalid')
        print ('cvalid_shape: ', cvalidX.shape)
        cvpreprocessed_data = run_preprocessor(sess, cvalidX, preprocess_graph, is_training=False)

        for epoch in range(0 ,epochs):
            tr_loss = []
            for batch_num in range(0, batch_size):
                batchX, batchY = load_batch_data(image_type='assessor_code', which_data='train_%s' % (batch_num))
                preprocessed_data = run_preprocessor(sess, batchX, preprocess_graph, is_training=True)
#                 print (epoch, batch_num)
                feed_dict = {computation_graph['inpX']: preprocessed_data}

                tr_ls, b_xent_ls, kl_ls,  _, lrate  = sess.run([computation_graph['loss'], computation_graph['b_xent_loss'], computation_graph['kl_loss'], computation_graph['optimizer'], computation_graph['learning_rate']], feed_dict=feed_dict)
                print (tr_ls, b_xent_ls, kl_ls)
                tr_loss.append(tr_ls)
                training_loss.append(tr_ls)
                learning_rate.append(lrate)

            feed_dict = {computation_graph['inpX']: cvpreprocessed_data}
            l_features, ts_ls = sess.run([computation_graph['latent_features'], computation_graph['loss']],
                                               feed_dict=feed_dict)
            print('l_features: ', l_features)
            print ('')
            test_loss.append(ts_ls)
            score = 0
            tot = 2
            rand_state = [223,431]
            for i in range(0,tot):
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=2, n_init=100, random_state=rand_state[i])
                kmeans = kmeans.fit(l_features)
                labels = kmeans.predict(l_features)
                centroids = kmeans.cluster_centers_
#                 print (labels)
#                 labels[labels == 0] = 6
#                 labels[labels == 1] = 8

                from conv_net.utils import Score
                scr = Score.accuracy(cvalidY, labels)
                score += max(scr, 1-scr)
            
            test_accuracy.append(score/tot)

            if (epoch%5)  == 0:
                print ('EPOCH = %s, Avg Train Loss = %s, Test Loss = %s, Cluster_accuracy = %s'%(str(epoch), str(np.sum(tr_loss)/len(tr_loss)), str(ts_ls), score/tot))
    return training_loss, test_loss, test_accuracy, learning_rate

training_loss, test_loss, test_accuracy, learning_rate = RUN(epochs=50, batch_size=15)

oj = Plot(rows=1, columns=3, fig_size=(40,7))

l_rate_df = pd.DataFrame(training_loss, columns=['training_loss'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'training_loss'})

tr_loss_df = pd.DataFrame(test_loss, columns=['test_loss'])
oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_loss'})

cv_loss_df = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_accuracy'})

epochs = 50
batch_size = 128
training_loss, test_loss, test_accuracy = RUN(epochs, batch_size,x_train, x_test, y_test)

oj = Plot(rows=1, columns=3, fig_size=(40,7))

l_rate_df = pd.DataFrame(training_loss, columns=['training_loss'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'training_loss'})

tr_loss_df = pd.DataFrame(test_loss, columns=['test_loss'])
oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_loss'})

cv_loss_df = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_accuracy'})

epochs = 50
batch_size = 128
training_loss, test_loss, test_accuracy, learning_rate = RUN(epochs, batch_size,x_train, x_test, y_test)

from plot import Plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

oj = Plot(rows=1, columns=4, fig_size=(40,7))

l_rate_df = pd.DataFrame(learning_rate, columns=['learning_rate'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'learning_rate'})

l_rate_df = pd.DataFrame(training_loss, columns=['training_loss'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'training_loss'})

tr_loss_df = pd.DataFrame(test_loss, columns=['test_loss'])
oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_loss'})

cv_loss_df = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_accuracy'})

epochs = 50
batch_size = 128
training_loss, test_loss, test_accuracy, learning_rate = RUN(epochs, batch_size,x_train, x_test, y_test)

oj = Plot(rows=1, columns=4, fig_size=(40,7))

l_rate_df = pd.DataFrame(learning_rate, columns=['learning_rate'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'learning_rate'})

l_rate_df = pd.DataFrame(training_loss, columns=['training_loss'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'training_loss'})

tr_loss_df = pd.DataFrame(test_loss, columns=['test_loss'])
oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_loss'})

cv_loss_df = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_accuracy'})

epochs = 50
batch_size = 128
training_loss, test_loss, test_accuracy, learning_rate = RUN(epochs, batch_size,x_train, x_test, y_test)

oj = Plot(rows=1, columns=4, fig_size=(40,7))

l_rate_df = pd.DataFrame(learning_rate, columns=['learning_rate'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'learning_rate'})

l_rate_df = pd.DataFrame(training_loss, columns=['training_loss'])
oj.vizualize(data=l_rate_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'training_loss'})

tr_loss_df = pd.DataFrame(test_loss, columns=['test_loss'])
oj.vizualize(data=tr_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_loss'})

cv_loss_df = pd.DataFrame(test_accuracy, columns=['test_accuracy'])
oj.vizualize(data=cv_loss_df, colX=None, colY=None, label_col=None, viz_type='line', params={'title':'test_accuracy'})



