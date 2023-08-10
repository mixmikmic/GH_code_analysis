#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import keras

#pretty plots
get_ipython().run_line_magic('matplotlib', 'inline')

#Load the index data
raw_data = pd.read_csv('index_data.csv', skiprows=1)
raw_data.set_index(['Date'])

#Relabeling the data
#relabeling the index data
index_close = raw_data.iloc[:,1:]
dates = raw_data.iloc[:,0]

#Display first 5 rows
raw_data.head(20)

#Dimensions
print("Number of data points:", index_close.shape[0])
print("Number of indices:", index_close.shape[1])

#lags, holding period
holding_period = 44 #5-trading days holding period

#Preprocess and transform to log-returns
#Calculate log returns.
daily_ret = np.log(index_close.shift(-44)/index_close)
#Remove first row
dates = dates.drop(dates.index[0:holding_period]).reset_index(drop=True)
daily_ret = daily_ret.drop(daily_ret.index[0:holding_period]).reset_index(drop=True)

daily_ret.head(20)

#Seperate the indices into 2 classes - lag or no_lag
no_lag = [0, 1, 2, 4, 5, 6, 9, 10]
lag = [i for i in range(0,daily_ret.shape[1]) if i not in no_lag]

#Processing the dataset by applying appropriate lags
lagged_data = daily_ret.iloc[:,lag].shift(1)
lagged_data = pd.concat([daily_ret.iloc[:,no_lag], lagged_data], axis=1)

#Removing the first row
lagged_data = lagged_data.drop(lagged_data.index[0]).reset_index(drop=True)
dates = dates.drop(dates.index[0]).reset_index(drop=True)

#Shifting HSI returns
lagged_data['Hang Seng Index'] = lagged_data['Hang Seng Index'].shift(-holding_period)
lagged_data = lagged_data.drop(lagged_data.index[-holding_period:]).reset_index(drop=True)
dates.drop(dates.index[-holding_period:]).reset_index(drop=True)

lagged_data.head(20)

#Calculate correlation
corr = lagged_data.corr()

#Plot the correlation heatmap
plt.figure(figsize=(20,20))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlations Heat Map', fontsize=20, fontweight='bold')
plt.show()

#Seperate the features and the labels
labels = lagged_data.iloc[:,0]
X = lagged_data.drop(daily_ret.columns[0], axis=1).reset_index(drop=True)

#Generate the labels from daily Hang Seng returns

#Set labels cutoff for defining Up/Down/Neutral states
labels_cutoff = 0.000627 #Try lower levels....0.05 seems a bit too high


#labels = labels - labels.shift(holding_period)
#labels = labels.drop(labels.index[0:holding_period]).reset_index(drop=True)
#X = X.drop(X.index[0:holding_period]).reset_index(drop=True)

y = np.empty(labels.shape)
y[labels < -labels_cutoff] = 0 #Label -1 for returns lower than -0.05%
y[labels > labels_cutoff] = 2 #Label 1 for returns greater than 0.05%
y[(labels <= labels_cutoff ) & (labels >= -labels_cutoff)] = 1 #Label 0 for the rest

#Import Keras module
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

#Import sklearn module
from sklearn.metrics import confusion_matrix

#Split into training set and test set
train_size = 0.8 #Again parameter to tune
val_size = 0.1

train_cut_index = int(np.floor(train_size * X.shape[0]))
val_cut_index = int(np.floor((train_size + val_size) * X.shape[0]))

X_train = np.array(X.iloc[0:(train_cut_index-1),:])
X_val = np.array(X.iloc[train_cut_index:(val_cut_index - 1),:])
X_test = np.array(X.iloc[val_cut_index:X.shape[0],:])

y_train = y[0:(train_cut_index-1)]
y_val = y[train_cut_index:val_cut_index-1]
y_test = y[val_cut_index:len(y)]

#Preproc the y_train and y_test using to_categorical function
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

#Prediction Period
print("Training Period Start:", dates[0])
print("Training Period End:", dates[train_cut_index-1])
print("Validation Period Start:", dates[train_cut_index])
print("Validation Period End:", dates[val_cut_index-1])
print("Test Period Start:", dates[val_cut_index])
print("Test Period End:", dates[len(dates)-1])
print("Number of data points in training set:", X_train.shape[0])
print("Number of data points in validation set:", X_val.shape[0])
print("Number of data points in test set:", X_test.shape[0])

#Reshape data for inputting into LSTM
timestep = 1
X_train = np.reshape(X_train, (X_train.shape[0], timestep, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], timestep, X_val.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], timestep, X_test.shape[1]))

y_train = np.reshape(y_train, (y_train.shape[0], timestep, y_train.shape[1]))
y_val = np.reshape(y_val, (y_val.shape[0], timestep, y_val.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0], timestep, y_test.shape[1]))

#Evaluation metrics
from keras import backend as K

#Define f-beta score
def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 0.6

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

#Define precision score
def precision(y_true, y_pred, threshold_shift=0):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)

    precision = tp / (tp + fp)

    return precision

def hit_rate(y_true, y_pred):
    
    #Calculate Hit Rate of the prediction
    true_class = K.argmax(y_true, axis = -1)
    pred_class = K.argmax(y_pred, axis = -1)
    

#Print model evaluation metrics, takes in model scores from training and test set
def print_metrics(model_score):
    print("Test Loss:", model_score[0])
    print("Test F-beta:", model_score[1])
    print("Test Precision:", model_score[2])
    print("Test Accuracy:", model_score[3])

#Define plot metrics - can add more metrics towards it

def plot_metrics(model):
       
    #Plotting Loss over Epoch
    plt.figure(1)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'], loc='upper left')
    
    #Plotting F-beta over Epoch
    plt.figure(2)
    plt.plot(model.history['fbeta'])
    plt.plot(model.history['val_fbeta'])
    plt.title('F-Beta Score, beta=0.6')
    plt.ylabel('F-Beta Score')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'], loc='upper left')
    
    #Plotting Precision over Epoch
    plt.figure(3)
    plt.plot(model.history['precision'])
    plt.plot(model.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'], loc='upper left')
    
    #Plotting Hit Rate over Epoch
    plt.figure(4)
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'], loc='upper left')
    
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Utility function to return class numbers from probabilities
def prob_to_class(pred):
    return(np.argmax(pred, axis=-1))

#Setting base LSTM Network Parameters
drop_out = 0.65
input_shape = X_train.shape
num_epoch = 100
loss_fcn = 'categorical_crossentropy'
learning_rate = 1
opt = optimizers.Nadam(lr=learning_rate)
alpha = 1
num_of_perceptron = np.int(0.5 * (X_train.shape[2] + 3)) * alpha

#Callbacks
saveModel = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True, mode='auto')
earlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_delta=0.0001)
callbacks = [saveModel, earlyStop]

get_ipython().run_cell_magic('time', '', "#Fit LSTM\nfrom keras.layers import LSTM\n\n#Setting up the instance\nlstm_model = Sequential()\n\n#Adding 1st LSTM layer\nlstm_model.add(LSTM(num_of_perceptron, input_shape=input_shape[1:], return_sequences=True, activation='tanh'))\nlstm_model.add(Dropout(drop_out))\n\n#Adding Output Layer\nlstm_model.add(Dense(3, activation='softmax'))\nlstm_model.summary()\n\n#Optimization\n#Define Optimizer, using Stochastic Gradient Decent\nlstm_model.compile(loss=loss_fcn, optimizer=opt, metrics=[fbeta, precision, 'accuracy'])\n\n#Fitting the model\nlstm = lstm_model.fit(X_train, y_train, epochs = num_epoch, verbose=1, validation_data=(X_val, y_val), shuffle=False, batch_size=1, callbacks=[earlyStop])\n\n#Evaluation\nlstm_score = lstm_model.evaluate(X_test, y_test)\n\n#Predict\nlstm_pred = lstm_model.predict(X_test, verbose=1)")

print_metrics(lstm_score)

plot_metrics(lstm)

cfm = confusion_matrix(prob_to_class(y_test), prob_to_class(lstm_pred)) 
plot_confusion_matrix(cfm, classes=['Down', 'Neutral', 'Up'], normalize=False)

lstm_pred

y_test









