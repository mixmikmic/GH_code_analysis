#Importing all the required Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pylab as plt

def DataLoading (mypath):
    print ("Loading the data")
    dataframe = pd.read_csv(mypath,header = None,engine = 'python',sep=",")
    return dataframe

def DataPreprocessing(mydataframe):
    
    # Dropping the duplicates
    recordcount = len(mydataframe)
    print ("Original number of records in the training dataset before removing duplicates is: " , recordcount)
    mydataframe.drop_duplicates(subset=None, inplace=True)  # Python command to drop duplicates
    newrecordcount = len(mydataframe)
    print ("Number of records in the training dataset after removing the duplicates is :", newrecordcount,"\n")

    #Dropping the labels to a different dataset which is used to train the recurrent neural network classifier
    df_X = mydataframe.drop(mydataframe.columns[41],axis=1,inplace = False)
    df_Y = mydataframe.drop(mydataframe.columns[0:41],axis=1, inplace = False)

    # Convert Categorial data to the numerical data for the efficient classification
    df_X[df_X.columns[1:4]] = df_X[df_X.columns[1:4]].stack().rank(method='dense').unstack()
    
    # Coding the normal as " 1 0" and attack as "0 1"
    df_Y[df_Y[41]!='normal.'] = 0
    df_Y[df_Y[41]=='normal.'] = 1
    #print (labels[41].value_counts())
    
    #converting input data into float which is requried in the future stage of building in the network
    df_X = df_X.loc[:,df_X.columns[0:41]].astype(float)

    # Normal is "1 0" and the abnormal is "0 1"
    df_Y.columns = ["y1"]
    df_Y.loc[:,('y2')] = df_Y['y1'] ==0
    df_Y.loc[:,('y2')] = df_Y['y2'].astype(int)
    

    
    return df_X,df_Y

def FeatureSelection(myinputX, myinputY):

    labels = np.array(myinputY).astype(int)
    inputX = np.array(myinputX)
    
    #Random Forest Model
    model = RandomForestClassifier(random_state = 0)
    model.fit(inputX,labels)
    importances = model.feature_importances_
    
    
    #Plotting the Features agains their importance scores
    indices = np.argsort(importances)[::-1]
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    plt.figure(figsize = (10,5))
    plt.title("Feature importances (y-axis) vs Features IDs(x-axis)")
    plt.bar(range(inputX.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(inputX.shape[1]), indices)
    plt.xlim([-1, inputX.shape[1]])
    plt.show()
    
    # Selecting top featueres which have higher importance values = here we can find "12" features
    #as we can see in the next step
    newX = myinputX.iloc[:,model.feature_importances_.argsort()[::-1][:12]]
   # Converting the X dataframe into tensors
    myX = newX.as_matrix()
    myY = labels

   
    return myX,myY

print ("Laoding the IDS Data")
data_path = "C:/Users/manp/Documents/IoT/Final/ApplicationLayer/ApplicationLayer.txt"
dataframe = DataLoading(data_path)

print ("Data Preprocessing of loaded IDS Data")
data_X, data_Y = DataPreprocessing(dataframe)

print ("Performing the Feature Selection on train data set")
reduced_X,reduced_Y = FeatureSelection(data_X,data_Y)

#Dividing the dataset into train and test datasets
# Out of 13051 samples = 80% samples as train data and 20% samples as test data

# Train features and Train Labels
train_X = reduced_X[:8000]
train_Y = reduced_Y[:8000]

#Test Features and Test Labels
test_X = reduced_X[8001:10000]
test_Y = reduced_Y[8001:10000]


print ("Train X shape is :", train_X.shape)
print ("Train Y shape is :", train_Y.shape)
print ("Test X shape is :", test_X.shape)
print ("Test Y shape is :", test_Y.shape)

# Before normalizing, the array of input features should be converted to a dataframe
semitrain_X = pd.DataFrame(train_X)
semitest_X = pd.DataFrame(test_X)
#Importing Scikit learn libraries
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Normalizing Train Data Features
scaler_traindata = scaler.fit(semitrain_X)
train_norm = scaler_traindata.transform(semitrain_X)
train_norm_X = pd.DataFrame(train_norm)

#Normalizing Test Data Features
scaler_testdata = scaler.fit(semitest_X)
test_norm = scaler_testdata.transform(semitest_X)
test_norm_X = pd.DataFrame(test_norm)

#Define Hyper Parameters for the model
learning_rate = 0.001
n_classes = 2
display_step = 100
input_features = train_X.shape[1] #No of selected features(columns)
training_cycles = 1000
time_steps = 5 # No of time-steps to backpropogate
hidden_units = 50 #No of LSTM units in a LSTM Hidden Layer

#Input Placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float64,shape = [None,time_steps,input_features], name = "x-input")
    y = tf.placeholder(tf.float64, shape = [None,n_classes],name = "y-input")
#Weights and Biases
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal([hidden_units,n_classes]),name = "layer-weights")

with tf.name_scope("biases"):
    b = tf.Variable(tf.random_normal([n_classes]),name = "unit-biases")

#Modify data with respect to the time steps count
def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

# Modifications to train data
train_data_X = pd.DataFrame(train_norm_X)
train_data_Y = pd.DataFrame(train_Y)
newtrain_X = rnn_data(train_data_X,time_steps,labels = False)
newtrain_Y = rnn_data(train_data_Y,time_steps,labels = True)

print ("Shape of new train X",newtrain_X.shape)
print ("Shape of new train Y",newtrain_Y.shape)

# Modifications to test data
test_data_X = pd.DataFrame(test_norm_X)
test_data_Y = pd.DataFrame(test_Y)
newtest_X = rnn_data(test_data_X,time_steps,labels = False)
newtest_Y = rnn_data(test_data_Y,time_steps,labels = True)

print ("Shape of new test X",newtest_X.shape)
print ("Shape of new test Y",newtest_Y.shape)

#Unstacking the inputs with time steps to provide the inputs in sequence
# Unstack to get a list of 'time_steps' tensors of shape (batch_size, input_features)
x_ = tf.unstack(x,time_steps,axis =1)

#Defining a single GRU cell
gru_cell = tf.contrib.rnn.GRUCell(hidden_units)

#GRU Output
with tf.variable_scope('MyGRUCel36'):
    gruoutputs,grustates = tf.contrib.rnn.static_rnn(gru_cell,x_,dtype=tf.float64)
    
#Linear Activation , using gru inner loop last output
output =  tf.add(tf.matmul(gruoutputs[-1],tf.cast(W,tf.float64)),tf.cast(b,tf.float64))

#Defining the loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Training the Model
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range (training_cycles):
    _,c = sess.run([optimizer,cost], feed_dict = {x:newtrain_X, y:newtrain_Y})
    
    if (i) % display_step == 0:
        print ("Cost for the training cycle : ",i," : is : ",sess.run(cost, feed_dict ={x :newtrain_X,y:newtrain_Y}))
correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy on the overall test set is :',accuracy.eval({x:newtest_X, y:newtest_Y}))

# tf.argmax = Returns the index with the largest value across axes of a tensor.
# Therefore, we are extracting the final labels => '1 0' = '1' = Normal (and vice versa)
# Steps to calculate the confusion matrix

pred_class = sess.run(tf.argmax(output,1),feed_dict = {x:newtest_X,y:newtest_Y})
labels_class = sess.run(tf.argmax(y,1),feed_dict = {x:newtest_X,y:newtest_Y})
conf = tf.contrib.metrics.confusion_matrix(labels_class,pred_class,dtype = tf.int32)
ConfM = sess.run(conf, feed_dict={x:newtest_X,y:newtest_Y})
print ("confusion matrix \n",ConfM)

#Plotting the Confusion Matrix
labels = ['Normal', 'Attack']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(ConfM)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculating Accuracy through another procedure
n = tf.cast(labels_class,tf.int64)
newaccuracy = tf.contrib.metrics.accuracy(pred_class,n)
print ("accuracy calcualted using tf.contrib", sess.run (newaccuracy, feed_dict = {x:newtest_X,y:newtest_Y}))

#Calculations performed manually for other metrics
TP = conf[0,0]
FN = conf [0,1]
FP = conf[1,0]
TN = conf[1,1]

AccConf = (TP+TN)/(TP+FP+TN+FN)
print ("Accuracy calculated manually through confusion matrix", sess.run (AccConf, feed_dict = {x:newtest_X,y:newtest_Y}))

# Precision
Precision = TP/(TP+FP)
print ("Precision \n",sess.run(Precision,feed_dict ={x:newtest_X,y:newtest_Y}))

#Recall also known as Sensitivity
Recall = TP/(TP+FN)
print ("Recall (DR) - Sensitivity [True Positive Rate]\n", sess.run(Recall,feed_dict={x:newtest_X,y:newtest_Y}))

# Specificity

Specificity = TN/(TN+FP)
print ("Specificity \n", sess.run(Specificity,feed_dict={x:newtest_X,y:newtest_Y}))

#F score
FScore = 2*((Precision*Recall)/(Precision+Recall))
print ("F1 Score is \n",sess.run(FScore,{x:newtest_X,y:newtest_Y}))

#False Alarm Rate
FAR = FP/(FP+TN)
print ("False Alarm Rate also known as False Positive Rate \n",sess.run(FAR,feed_dict ={x:newtest_X,y:newtest_Y}))

#Efficiency
Efficiency = Recall/FAR
print("Efficincy is \n",sess.run(Efficiency,feed_dict = {x:newtest_X,y:newtest_Y}))





