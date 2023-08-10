# Author: Hamaad Musharaf Shah.
from PIL import Image

from six.moves import range

import os
import math
import sys
import importlib

import numpy as np

import pandas as pd

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as bkend
from keras.datasets import cifar10, mnist
from keras import layers
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, convolutional, pooling, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import metrics
from keras.models import Model
from keras.utils.generic_utils import Progbar

import tensorflow as tf
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt

from plotnine import *
import plotnine

from gan_keras.dcgan import DeepConvGenAdvNet, DeeperConvGenAdvNet, DeepConvGenAdvNetInsurance

get_ipython().magic("matplotlib inline")

os.environ["KERAS_BACKEND"] = "tensorflow"
importlib.reload(bkend)

print(device_lib.list_local_devices())

mnist = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])
y_train = y_train.ravel()
y_test = y_test.ravel()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255.0
x_test /= 255.0

scaler_classifier = MinMaxScaler(feature_range=(0.0, 1.0))
logistic = linear_model.LogisticRegression(random_state=666, verbose=1)
lb = LabelBinarizer()
lb = lb.fit(y_train.reshape(y_train.shape[0], 1))

dcgan = DeepConvGenAdvNet(batch_size=100,
                          iterations=10000,
                          z_size=2)

pipe_dcgan = Pipeline(steps=[("DCGAN", dcgan),
                             ("scaler_classifier", scaler_classifier),
                             ("classifier", logistic)])
pipe_dcgan = pipe_dcgan.fit(x_train, y_train)

acc_dcgan = pipe_dcgan.score(x_test, y_test)

print("The accuracy score for the MNIST classification task with DCGAN: %.6f%%." % (acc_dcgan * 100))

n = 50
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-1.0, 1.0, n)
grid_y = np.linspace(-1.0, 1.0, n)

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = pipe_dcgan.named_steps["DCGAN"].generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(20, 20))
plt.imshow(figure, cmap="Greys")
plt.title("Deep Convolutional Generative Adversarial Network (DCGAN) with a 2-dimensional latent manifold\nGenerating new images on the 2-dimensional latent manifold", fontsize=20)
plt.xlabel("Latent dimension 1", fontsize=24)
plt.ylabel("Latent dimension 2", fontsize=24)
plt.savefig(fname="DCGAN_Generated_Images.png")

deepercgan = DeeperConvGenAdvNet(batch_size=100,
                                 iterations=10000,
                                 z_size=2)

pipe_deepercgan = Pipeline(steps=[("DeeperCGAN", deepercgan),
                                  ("scaler_classifier", scaler_classifier),
                                  ("classifier", logistic)])
pipe_deepercgan = pipe_deepercgan.fit(x_train, y_train)

acc_deepercgan = pipe_deepercgan.score(x_test, y_test)

print("The accuracy score for the MNIST classification task with Deeper CGAN: %.6f%%." % (acc_deepercgan * 100))

n = 50
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = pipe_deepercgan.named_steps["DeeperCGAN"].generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(20, 20))
plt.imshow(figure, cmap="Greys")
plt.title("Deeper Convolutional Generative Adversarial Network (DCGAN) with a 2-dimensional latent manifold\nGenerating new images on the 2-dimensional latent manifold", fontsize=20)
plt.xlabel("Latent dimension 1", fontsize=24)
plt.ylabel("Latent dimension 2", fontsize=24)
plt.savefig(fname="DeeperCGAN_Generated_Images.png")

claim_risk = pd.read_csv(filepath_or_buffer="data/claim_risk.csv")
claim_risk.drop(columns="policy.id", axis=1, inplace=True)
claim_risk = np.asarray(claim_risk).ravel()

transactions = pd.read_csv(filepath_or_buffer="data/transactions.csv")
transactions.drop(columns="policy.id", axis=1, inplace=True)

n_policies = 1000
n_transaction_types = 3
n_time_periods = 4

transactions = np.reshape(np.asarray(transactions), (n_policies, n_time_periods * n_transaction_types))

X_train, X_test, y_train, y_test = train_test_split(transactions, claim_risk, test_size=0.3, random_state=666)

min_X_train = np.apply_along_axis(func1d=np.min, axis=0, arr=X_train)
max_X_train = np.apply_along_axis(func1d=np.max, axis=0, arr=X_train) 
range_X_train = max_X_train - min_X_train + sys.float_info.epsilon
X_train = (X_train - min_X_train) / range_X_train
X_test = (X_test - min_X_train) / range_X_train
transactions = (transactions - min_X_train) / range_X_train

X_train = np.reshape(np.asarray(X_train), (X_train.shape[0], n_time_periods, n_transaction_types, 1))
X_test = np.reshape(np.asarray(X_test), (X_test.shape[0], n_time_periods, n_transaction_types, 1))
transactions = np.reshape(np.asarray(transactions), (n_policies, n_time_periods, n_transaction_types, 1))

dcgan_ins = DeepConvGenAdvNetInsurance(batch_size=50,
                                       iterations=5000,
                                       z_size=2)

pipe_dcgan_ins = Pipeline(steps=[("DCGANIns", dcgan_ins),
                                 ("scaler_classifier", scaler_classifier),
                                 ("classifier", logistic)])

pipe_dcgan_ins = pipe_dcgan_ins.fit(X_train, y_train)

auroc_dcgan_ins = roc_auc_score(y_true=y_test,
                                y_score=pipe_dcgan_ins.predict_proba(X_test)[:, 1], 
                                average="weighted")

print("The AUROC score for the insurance classification task with DCGAN: %.6f%%." % (auroc_dcgan_ins * 100))

n = 50
figure = np.zeros((n_time_periods * n, n_transaction_types * n))
lattices = []
actual_lattices = []
grid_x = np.linspace(-1.0, 1.0, n)
grid_y = np.linspace(-1.0, 1.0, n)

counter = 0
for i in range(transactions.shape[0]):
    actual_lattices.append(pd.DataFrame(transactions[i, :, :, 0], columns=["Paid", "Reserves", "Recoveries"]))
    actual_lattices[counter]["Unit"] = counter
    actual_lattices[counter]["Time"] = ["Period 1", "Period 2", "Period 3", "Period 4"]
    actual_lattices[counter]["Type"] = "Actual"
    counter += 1

counter = 0
for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = pipe_dcgan_ins.named_steps["DCGANIns"].generator.predict(z_sample)
        lattice = x_decoded[0].reshape(n_time_periods, n_transaction_types)
        lattices.append(pd.DataFrame(lattice, columns=["Paid", "Reserves", "Recoveries"]))
        lattices[counter]["Unit"] = counter
        lattices[counter]["Time"] = ["Period 1", "Period 2", "Period 3", "Period 4"]
        lattices[counter]["Type"] = "Generated"
        counter += 1
        figure[i * n_time_periods: (i + 1) * n_time_periods, j * n_transaction_types: (j + 1) * n_transaction_types] = lattice

plt.figure(figsize=(20, 30))
plt.imshow(figure, cmap="Greys")
plt.title("Deep Convolutional Generative Adversarial Network (DCGAN) with a 2-dimensional latent manifold for the insurance data\nGenerating new transactions data on the 2-dimensional latent manifold", fontsize=20)
plt.xlabel("Latent dimension 1", fontsize=24)
plt.ylabel("Latent dimension 2", fontsize=24)
plt.savefig(fname="DCGAN_Generated_Lattices.png")

tmp_act = pd.melt(pd.concat(actual_lattices, axis=0), id_vars=["Unit", "Time", "Type"])
tmp_gen = pd.melt(pd.concat(lattices, axis=0), id_vars=["Unit", "Time", "Type"])
plot_out = pd.concat([tmp_act, tmp_gen], axis=0)

plotnine.options.figure_size = (21, 15)
dens_plot = ggplot(plot_out) + geom_density(aes(x="value", fill="factor(Type)"), alpha=0.5, color="black") + xlab("Actual and generated data: Note that both lie on the [0, 1] interval") + ylab("Density") + facet_wrap(facets=["Time", "variable"], scales="free_y", ncol=3) + ggtitle("Deep Convolutional Generative Adversarial Network (DCGAN) for the insurance data\nA simple sanity check comparison of actual and generated transactions distributions") + theme(legend_position="bottom") + theme_matplotlib()
print(dens_plot)

dens_plot.save(filename="Dens_Plots_Actual_Generated_Lattices.png")

should_we_run = False
if should_we_run:
    i = j = 0
    plt.figure(figsize=(20, 30))
    plt.imshow(figure[i * n_time_periods: (i + 1) * n_time_periods, j * n_transaction_types: (j + 1) * n_transaction_types], cmap="Greys")
    plt.title("Showing an example of a generated transactions lattice", fontsize=24)
    plt.xlabel("Latent dimension 1", fontsize=24)
    plt.ylabel("Latent dimension 2", fontsize=24)
    plt.savefig(fname="DCGAN_Generated_Lattice_Example_Plotted.png")
    pd.DataFrame(lattices[i])

