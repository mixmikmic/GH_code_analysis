import os
import pandas as pd
import math
import gzip
import numpy as np
import scipy.stats as stats
from scipy.sparse import csc_matrix, csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from sklearn.model_selection import train_test_split as sklearn_train_test_split
import seaborn as sns; sns.set(style="white", color_codes=True)
import warnings
import tensorflow as tf
import TrainingSparseBiasonly as tdb
import TrainingSparseVector as tdv
import TrainingSparseIT as tdit
import TrainingSparseUT as tdut
import TrainingSparseUTDay as tdutday
import TrainingSparseUITDay as tdiutday
import TrainingSparseUIVectorTDay as tdiuvtday
import TrainingSparseUIVectorT_Implicit as tall
import time
from DataProcessing import getBins, train_test_split,PreProcessAmazonDF, convert_csr_to_sparse_tensor_inputs, getMeanDaybyUser, get_df_base_loss
from DataProcessing import getImplicitDF, get_base_loss, parse, getDF, getcsvDF, convert_to_sparse_tensor_inputs, getMeanDay,  patch_with_value, load_from_raw_df, getUserRatedItemCount, getUserRatedItemCountNonUnique
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#df = getDF('data\\reviews_Patio_Lawn_and_Garden_5.json.gz')
#df = getDF('reviews_Movies_and_TV_10.json.gz')
#df = getDF('data\\reviews_CDs_and_Vinyl_10.json.gz')
#df= getDF('data\\reviews_Electronics_10.json.gz')
df = getDF('data\\reviews_Beauty_10.json.gz')
#df = getDF('data\\reviews_Office_Products_10.json.gz')
#df = getDF("data\\reviews_Home_and_Kitchen_10.json.gz")
#df = getDF("data\\reviews_Video_Games_10.json.gz")


df = PreProcessAmazonDF(df, bin_count = tdut.Config.item_bin_size)
df.head(5)


df = df.drop("reviewerName", 1)
df = df.drop("reviewText",1)
df = df.drop("summary", 1)


df.head(10)

train_df, test_df = sklearn_train_test_split(df, test_size=0.2, random_state = 42)
train_df, dev_df = sklearn_train_test_split(train_df, test_size=0.25, random_state = 38)

implicit_train_df = getImplicitDF(train_df)

num_user = df.groupby("userID")["userID"].unique().count()
num_item = df.groupby("itemID")["itemID"].unique().count()
print(num_user, num_item)

average_rank = df["overall"].mean()

fig = plt.figure(figsize=(10,10))

mean_rating_of_user = df.groupby('reviewerID').apply(lambda x: x['overall'].mean())
ax = fig.add_subplot(322)
ax.hist(mean_rating_of_user)
ax.set_xlabel('mean rating given by each user', fontsize=15)

plt.tight_layout()
plt.show()

average_rank = df["overall"].mean()
fig = plt.figure(figsize=(10,10))

# mean ratings from each user
mean_rating_of_item = df.groupby('product').apply(lambda x: x['overall'].mean())
ax = fig.add_subplot(322)
ax.hist(mean_rating_of_user)
ax.set_xlabel('mean rating by each product', fontsize=15)

plt.tight_layout()
plt.show()

def train_by_df(from_model, train_df, dev_df, test_df, num_user, num_item, debug = True, implicit_df = None):
    print (80 * "=")
    print ("INITIALIZING")
    print( 80 * "=")
    config = from_model.Config()
    config.n_items = num_item 
    config.n_users = num_user
    config.maxday_cat_code = max(train_df["TDayCat"].values)
    
    if not os.path.exists('data\\weights\\'):
        os.makedirs('.\\data\\weights\\')

    with tf.Graph().as_default():
        print ("Building model...",)
        start = time.time()
        model = from_model.RecommandationModel(config) 
        

        init = tf.global_variables_initializer()

        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            #recommand.session = session
            session.run(init)

            print( 80 * "=")
            print( "TRAINING")
            print( 80 * "=")
            model.fit(session, saver, train_df, dev_df) #, implicit_df)
            print("Done!")
            print ("took {:.2f} seconds\n".format(time.time() - start))
            
            if not debug:
                print (80 * "=")
                print ("TESTING")
                print( 80 * "=")
                print ("Restoring the best model weights found on the dev set")
                #saver.restore(session, model.config.weight_filename)
                print ("Final evaluation on test set",)
                mean_rank = train_df["overall"].mean() #global rank mean
        
                mean_u_day = getMeanDaybyUser(train_df)
                test_loss = model.run_epoch(session, train_df, mean_rank, mean_u_day, test_df) 
                
                print ("- test l2 loss: {:.5f}", test_loss)

                print ("Done!")
                
                return test_loss

def train_by_df_timeVCDplus(from_model, train_df, dev_df, test_df, num_user, num_item, debug = True, implicit_df = None):
    print (80 * "=")
    print ("INITIALIZING")
    print( 80 * "=")
    config = from_model.Config()
    config.n_items = num_item 
    config.n_users = num_user
    config.maxday_cat_code = max(train_df["TDayCat"].values)
    
    if not os.path.exists('data\\weights\\'):
        os.makedirs('.\\data\\weights\\')

    with tf.Graph().as_default():
        print ("Building model...",)
        start = time.time()
        model = from_model.RecommandationModel(config) 
        

        init = tf.global_variables_initializer()

        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            #recommand.session = session
            session.run(init)

            print( 80 * "=")
            print( "TRAINING")
            print( 80 * "=")
            model.fit(session, saver, train_df, dev_df, implicit_df)
            print("Done!")
            print ("took {:.2f} seconds\n".format(time.time() - start))
            
            if not debug:
                print (80 * "=")
                print ("TESTING")
                print( 80 * "=")
                print ("Restoring the best model weights found on the dev set")
                #saver.restore(session, model.config.weight_filename)
                print ("Final evaluation on test set",)
                mean_rank = train_df["overall"].mean() #global rank mean
        
                mean_u_day = getMeanDaybyUser(train_df)
                test_loss = model.run_epoch(session, train_df, mean_rank, mean_u_day, test_df, implicit_df) 
                
                print ("- test l2 loss: {:.5f}", test_loss)

                print ("Done!")
                
                return test_loss

loss_dict={}
loss_dict["Base"] = get_df_base_loss(train_df, dev_df, test_df)[2]

loss_dict["TALL"] = train_by_df_timeVCDplus(tall, train_df, dev_df, test_df, num_user, num_item, debug = False, implicit_df = implicit_train_df )

loss_dict["VTDay"] = train_by_df(tdiuvtday, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["IUDay"] = train_by_df(tdiutday, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["UserDay"] = train_by_df(tdutday, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["UserTime"] = train_by_df(tdut, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["ItemTime"] = train_by_df(tdit, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["Vector"] = train_by_df(tdv, train_df, dev_df, test_df, num_user, num_item, debug = False)

loss_dict["Biasonly"] = train_by_df(tdb, train_df, dev_df, test_df, num_user, num_item, debug = False)

N = len(loss_dict.values())
x = list(loss_dict.values()) 

ind = np.arange(N)  
width = 0.35       # the width of the bars

fig, ax = plt.subplots(figsize=(9, 7))
g1 = ax.bar(ind, x, width, color='b')
g2 = ax.plot(x, 'r')

# add some text for labels, title and axes ticks
ax.set_ylabel('L2 Loss', fontsize=15)
ax.set_title('Model Comparison', fontsize=15)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(list(loss_dict.keys())) 
ax.tick_params(labelsize=10)
ax.set_ylim([0.65, max(x) + 0.1])

print(loss_dict)

