import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import pandas as pd
print(pd.__version__)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

import keras
print(keras.__version__)

# df = pd.read_csv('./insurance-customers-300.csv', sep=';')
df = pd.read_csv('./insurance-customers-1500.csv', sep=';')

y=df['group']

df.drop('group', axis='columns', inplace=True)

X = df.as_matrix()

df.describe()

# ignore this, it is just technical code
# should come from a lib, consider it to appear magically 
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_print = ListedColormap(['#AA8888', '#004000', '#FFFFDD'])
cmap_bold = ListedColormap(['#AA4444', '#006000', '#AAAA00'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#FFFFDD'])
font_size=25

def meshGrid(x_data, y_data):
    h = 1  # step size in the mesh
    x_min, x_max = x_data.min() - 1, x_data.max() + 1
    y_min, y_max = y_data.min() - 1, y_data.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return (xx,yy)
    
def plotPrediction(clf, x_data, y_data, x_label, y_label, colors, title="", mesh=True, fixed=None, fname=None, print=False):
    xx,yy = meshGrid(x_data, y_data)
    plt.figure(figsize=(20,10))

    if clf and mesh:
        grid_X = np.array(np.c_[yy.ravel(), xx.ravel()])
        if fixed:
            fill_values = np.full((len(grid_X), 1), fixed)
            grid_X = np.append(grid_X, fill_values, axis=1)
        Z = clf.predict(grid_X)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    if print:
        plt.scatter(x_data, y_data, c=colors, cmap=cmap_print, s=200, marker='o', edgecolors='k')
    else:
        plt.scatter(x_data, y_data, c=colors, cmap=cmap_bold, s=80, marker='o', edgecolors='k')
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    if fname:
        plt.savefig(fname)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

X_train_kmh_age = X_train[:, :2]
X_test_kmh_age = X_test[:, :2]
X_train_2_dim = X_train_kmh_age
X_test_2_dim = X_test_kmh_age

# tiny little pieces of feature engeneering
from keras.utils.np_utils import to_categorical

num_categories = 3

y_train_categorical = to_categorical(y_train, num_categories)
y_test_categorical = to_categorical(y_test, num_categories)

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout

inputs = Input(name='input', shape=(2, ))
x = Dense(100, name='hidden1', activation='relu')(inputs)
x = Dense(100, name='hidden2', activation='relu')(x)
x = Dense(100, name='hidden3', activation='relu')(x)
predictions = Dense(3, name='softmax', activation='softmax')(x)
model = Model(input=inputs, output=predictions)

# loss function: http://cs231n.github.io/linear-classify/#softmax
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

get_ipython().magic('time model.fit(X_train_2_dim, y_train_categorical, epochs=1000, verbose=0, batch_size=100)')
# %time model.fit(X_train_2_dim, y_train_categorical, epochs=1000, validation_split=0.2, verbose=0, batch_size=100)

plotPrediction(model, X_train_2_dim[:, 1], X_train_2_dim[:, 0], 
               'Age', 'Max Speed', y_train,
                title="Train Data Max Speed vs Age with Classification")

train_loss, train_accuracy = model.evaluate(X_train_2_dim, y_train_categorical, batch_size=100)
train_accuracy

plotPrediction(model, X_test_2_dim[:, 1], X_test_2_dim[:, 0], 
               'Age', 'Max Speed', y_test,
                title="Test Data Max Speed vs Age with Prediction")

test_loss, test_accuracy = model.evaluate(X_test_2_dim, y_test_categorical, batch_size=100)
test_accuracy

drop_out = 0.15

inputs = Input(name='input', shape=(3, ))
x = Dense(100, name='hidden1', activation='relu')(inputs)
x = Dropout(drop_out)(x)
x = Dense(100, name='hidden2', activation='relu')(x)
x = Dropout(drop_out)(x)
x = Dense(100, name='hidden3', activation='relu')(x)
x = Dropout(drop_out)(x)
# x = Dense(100, name='hidden4', activation='sigmoid')(x)
# x = Dropout(drop_out)(x)
# x = Dense(100, name='hidden5', activation='sigmoid')(x)
# x = Dropout(drop_out)(x)
predictions = Dense(3, name='softmax', activation='softmax')(x)
model = Model(input=inputs, output=predictions)

# loss function: http://cs231n.github.io/linear-classify/#softmax
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
get_ipython().magic('time model.fit(X_train, y_train_categorical, epochs=1000, verbose=0, batch_size=100)')
# %time model.fit(X_train, y_train_categorical, epochs=1000, validation_split=0.2, verbose=0, batch_size=100)

train_loss, train_accuracy = model.evaluate(X_train, y_train_categorical, batch_size=100)
train_accuracy

test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, batch_size=100)
test_accuracy

kms_per_year = 15
plotPrediction(model, X_test[:, 1], X_test[:, 0],
               'Age', 'Max Speed', y_test,
                fixed = kms_per_year,
                title="Test Data Max Speed vs Age with Prediction, 15 km/year",
                fname='cnn.png')

kms_per_year = 50
plotPrediction(model, X_test[:, 1], X_test[:, 0],
               'Age', 'Max Speed', y_test,
                fixed = kms_per_year,
                title="Test Data Max Speed vs Age with Prediction, 50 km/year")

prediction = model.predict(X)
y_pred = np.argmax(prediction, axis=1)

y_true = y

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

import seaborn as sns
sns.heatmap(cm, annot=True, cmap="YlGnBu")
figure = plt.gcf()
figure.set_size_inches(10, 10)
ax = figure.add_subplot(111)
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')

inputs = Input(name='input', shape=(3, ))
x = Dense(80, name='hidden1', activation='relu')(inputs)
x = Dense(80, name='hidden2', activation='relu')(x)
x = Dense(80, name='hidden3', activation='relu')(x)
predictions = Dense(3, name='softmax', activation='softmax')(x)
model = Model(input=inputs, output=predictions)

# loss function: http://cs231n.github.io/linear-classify/#softmax
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

get_ipython().magic('time model.fit(X_train, y_train_categorical, epochs=1000, verbose=0, batch_size=100)')
# %time model.fit(X_train, y_train_categorical, epochs=1000, validation_split=0.2, verbose=0, batch_size=100)

train_loss, train_accuracy = model.evaluate(X_train, y_train_categorical, batch_size=100)
print(train_accuracy)

test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, batch_size=100)
print(test_accuracy)

get_ipython().system('rm -rf tf')

import os
from keras import backend as K

# K.clear_session()

K.set_learning_phase(0)

export_path_base = 'tf'
export_path = os.path.join(
  tf.compat.as_bytes(export_path_base),
  tf.compat.as_bytes("1"))

sess = K.get_session()

classification_inputs = tf.saved_model.utils.build_tensor_info(model.input)
classification_outputs_scores = tf.saved_model.utils.build_tensor_info(model.output)

from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def

signature = predict_signature_def(inputs={'inputs': model.input},
                                  outputs={'scores': model.output})

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

builder.add_meta_graph_and_variables(
      sess, 
     tags=[tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
      })

builder.save()

get_ipython().system('ls -lhR tf')

get_ipython().system('saved_model_cli show --dir tf/1 --tag_set serve --signature_def serving_default')

# 0: red
# 1: green
# 2: yellow

get_ipython().system("saved_model_cli run --dir tf/1 --tag_set serve --signature_def serving_default --input_exprs 'inputs=[[160.0,47.0,15.0]]'")

get_ipython().system('cat sample_insurance.json')

# https://cloud.google.com/ml-engine/docs/deploying-models

# Copy model to bucket
# gsutil cp -R tf/1 gs://booster_bucket

# create model and version at https://console.cloud.google.com/mlengine

# try ouy deployed
# gcloud ml-engine predict --model=booster --version=v1 --json-instances=./sample_insurance.json
# SCORES
# [0.003163766348734498, 0.9321494698524475, 0.06468681246042252]
# [2.467862714183866e-08, 1.2279541668431052e-14, 1.0]

