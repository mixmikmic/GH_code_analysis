from __future__ import print_function
import keras
import operator
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Lambda
from keras.activations import relu
from keras import backend as K

import tensorflow as tf

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'notebook')

#set up data for training
batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model for training with sigmoid activation
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5),
                 input_shape=input_shape))
model.add(Activation(relu))#, max_value=1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, (5, 5)))
model.add(Activation(relu))#, max_value=1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation(relu))#, max_value=1)))
model.add(Dense(84))
model.add(Activation(relu))#, max_value=1)))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#show summary of our model
model.summary()

#train and test our model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for layer in model.layers:
    try:
        print(np.amax(layer.get_weights()[0]), np.amin(layer.get_weights()[0]),
              np.amax(layer.get_weights()[1]), np.amin(layer.get_weights()[1]))
    except:
        continue

val = K.variable(256, name="val")
def lfunc(x):
    return K.round(x/val)

model_discrete = Sequential()
model_discrete.add(Conv2D(6, kernel_size=(5, 5),
                 input_shape=input_shape, activation='relu', use_bias=False))
model_discrete.add(Lambda(lfunc))
#model_discrete.add(Activation(lambda x: relu(x, max_value=65200)))
model_discrete.add(MaxPooling2D(pool_size=(2,2)))
model_discrete.add(Conv2D(16, (5, 5), activation='relu', use_bias=False))
model_discrete.add(Lambda(lfunc))
#model_discrete.add(Activation(lambda x: relu(x, max_value=65200)))
model_discrete.add(MaxPooling2D(pool_size=(2, 2)))
model_discrete.add(Flatten())
model_discrete.add(Dense(120, activation='relu'))
model_discrete.add(Lambda(lfunc))
#model_discrete.add(Activation(lambda x: relu(x, max_value=65200)))
model_discrete.add(Dense(84, activation='relu'))
model_discrete.add(Lambda(lfunc))
#model_discrete.add(Activation(lambda x: relu(x, max_value=65200)))
model_discrete.add(Dense(num_classes))

#manually evaluate our model on a test set
def evaluate_model(model, test_x, test_y):
    num_correct = 0
    for inp, outp in zip(test_x, test_y):
        pred = model.predict(np.reshape(inp, (1, 28, 28, 1)))
        max_index, max_value = max(enumerate(pred[0]), key=operator.itemgetter(1))
        if int(outp[max_index]) == 1:
            num_correct += 1
    return num_correct #return # correctly predicted

#magic function that applies an operation to every element in a numpy ndarray
def mod_ndarray(array, operation):
    if array.ndim == 1:
        return [operation(x) for x in array]
    else:
        return [mod_ndarray(x, operation) for x in array]

#turns a value from 0 to 1 into uint(8)
def mult_255(val):
    return round(val*255.0)

for i in range(0, len(model.layers)):        
    try:
        model_discrete.layers[i].set_weights([np.asarray(mod_ndarray(model.layers[i].get_weights()[0], mult_255)), np.asarray(mod_ndarray(model.layers[i].get_weights()[1], mult_255))])
    except:
        try:
            model_discrete.layers[i].set_weights([np.asarray(mod_ndarray(model.layers[i].get_weights()[0], mult_255))])
        except:
            continue
        
num_correct = evaluate_model(model_discrete, x_test, y_test)
print(num_correct, "predicted correctly")
print(num_correct/100, "% predicted correctly")

fh = open("inputs.txt", "w")
for inp in x_test.flatten():
    fh.write(str(inp)+"\n")
fh.close()

fh = open("flat_weights_float.txt", "w")
for layer in model.layers:
    wgt = layer.get_weights()
    if wgt:
        weights = wgt[0]
        bias = wgt[1]
        fh.write(layer.get_config()['name']+"\n")
        for s in weights.T.shape:
            fh.write(str(s) + " ")
        fh.write("\n")
        for weight in weights.T.flatten():
            fh.write(str(weight)+" ")
        fh.write("\n")
        for s in bias.shape:
            fh.write(str(s) + " ")
        fh.write("\n")
        for term in bias.flatten():
            fh.write(str(term)+" ")
        fh.write("\n\n")
fh.close()

fh = open("flat_weights_discrete.txt", "w")
for i in range(0, len(model_discrete.layers)):
    layer = model_discrete.layers[i]
    wgt = layer.get_weights()
    if wgt:
        weights = wgt[0]
        fh.write(layer.get_config()['name']+"\n")
        for s in weights.shape:
            fh.write(str(s) + " ")
        fh.write("\n")
        if 'conv' in layer.get_config()['name']:
            for matrix in weights.T:
                for weight in matrix[0].T.flatten():
                    fh.write(str(weight)+" ")
        else:
            for weight in weights.T.flatten():
                fh.write(str(weight)+" ")
        fh.write("\n")
        if 'conv' in layer.get_config()['name']:
            bias = model.layers[i].get_weights()[1]
            for s in bias.shape:
                fh.write(str(s) + " ")
            fh.write("\n")
            for s in bias.flatten():
                fh.write(str(0) + " ")
        else:
            bias = wgt[1]
            for s in bias.shape:
                fh.write(str(s) + " ")
            fh.write("\n")
            for term in bias.flatten():
                fh.write(str(term)+" ")
        fh.write("\n\n")
fh.close()

print(model_discrete.layers[7].get_weights()[0])

print(len(model_discrete.layers[7].get_weights()[0]))

res = model.predict(x_test)

print(res)

def keras_get_layer_output(model, layer, test_input):
    """
    Helper method, gives the output matrix from a Keras layer
    """
    get_layer_output = K.function([model.layers[0].input],
                                  [layer.output])
    return get_layer_output([test_input])[0]

np.set_printoptions(suppress=True, threshold=100000)

maxmin = []
for tst in x_test[0:1]:
    testin = np.reshape(tst, (1, 28, 28, 1))
    print("input = ")
    print(tst)
    i = 1
    for layer in model_discrete.layers:
        print("layer", i,  "out:")
        out = keras_get_layer_output(model_discrete, layer, testin)
        print(out)
        maxmin.append(np.amax(out))
        maxmin.append(np.amin(out))
        i+=1

print(max(maxmin), min(maxmin))

model_discrete.save("model_discrete.h5")

testin = np.reshape(x_test[0], (1, 28, 28, 1))
out = keras_get_layer_output(model_discrete, model_discrete.layers[0], testin)

print(out.shape)

weights1 = model_discrete.layers[0].get_weights()[0]

print(weights1.T)

for i in range(0, 28):
    for j in range(0, 28):
        print(x_test[0][i][j][0], end=' ')
    print()

for i in range(0, 24):
    for j in range(0, 24):
        print(out.T[0][i][j][0], end=' ')
    print()

model_discrete.layers[0].get_weights()[0].shape

model_discrete.layers[3].get_weights()[0].shape

fh = open("flat_weights_discrete.txt", "w")
for i in range(0, len(model_discrete.layers)):
    layer = model_discrete.layers[i]
    wgt = layer.get_weights()
    if wgt:
        weights = wgt[0]
        fh.write(layer.get_config()['name']+"\n")
        for s in weights.shape:
            fh.write(str(s) + " ")
        fh.write("\n")
        if 'conv' in layer.get_config()['name']:
            for j in range(0, len(weights[0][0])):
                for i in range(0, len(weights[0][0][0])):
                    for k in range(0, len(weights[0])):
                        for l in range(0, len(weights)):
                            fh.write(str(weights[l][k][j][i]) + " ")
        else:
            for weight in weights.T.flatten():
                fh.write(str(weight)+" ")
        fh.write("\n")
        if 'conv' in layer.get_config()['name']:
            #bias = model.layers[i].get_weights()[1]
            #for s in bias.shape:
            #    fh.write(str(s) + " ")
            #fh.write("\n")
            #for s in bias.flatten():
            #    fh.write(str(0) + " ")
            pass
        else:
            bias = wgt[1]
            for s in bias.shape:
                fh.write(str(s) + " ")
            fh.write("\n")
            for term in bias.flatten():
                fh.write(str(term)+" ")
        fh.write("\n\n")
fh.close()

inp = [0 for i in range(0, 28*28)]
inp[58] = 100
inp = np.asarray(inp)
inp = np.reshape(inp, (1, 28, 28, 1))
for i in range(0, len(model_discrete.layers)):
    outp = keras_get_layer_output(model_discrete, model_discrete.layers[i], inp)
    print(outp.T.flatten())

print("{0", end="")
for val in range(0, 28*28):
    print(","+str(val), end="")
print("}")

outp = keras_get_layer_output(model_discrete, model_discrete.layers[8], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp.shape)

real_wgt_dense1 = model_discrete.layers[7].get_weights()

s = [0 for i in range(0, 240)]
s.extend([0 for i in range(0, 30480)])
s = np.asarray(s).reshape(256, 120)
for i in range(0 , 256):
    s[i][0] = 1
print(s.shape)
model_discrete.layers[7].set_weights([s, np.asarray([0 for i in range(0, 120)])])
outp = keras_get_layer_output(model_discrete, model_discrete.layers[7], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp)
outp2 = keras_get_layer_output(model_discrete, model_discrete.layers[6], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp2)
outp3 = keras_get_layer_output(model_discrete, model_discrete.layers[5], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp3.T)

#print(outp.T[0])
for k in range(0, 16):
    for i in range(0, 4):
        for j in range(0, 4):
            print(outp.T[k][j][i][0], end=" ")
        print()
    print()
    print()
    for i in range(0, 4):
        for j in range(0, 4):
            print(outp.T[k][i][j][0], end=" ")
        print()
    print()
    print()

fh = open("/Users/thomasboser/Documents/NIPS-2017/animation_infiles/fpga_output.txt", "w")
for testin in x_test:
    tst = np.reshape(testin, (1, 28, 28, 1))
    out = model_discrete.predict(tst)
    for val in out.flatten():
        fh.write(str(val)+" ")
    fh.write(str((np.random.randn()+1000.0)/1500000.0)+"\n")
fh.close()

model_discrete.layers[0].get_weights()[0].T

weights = model_discrete.layers[3].get_weights()[0]

for i in range(0, len(weights[0][0][0])):
    for j in range(0, len(weights[0][0])):
        for k in range(0, len(weights[0])):
            for l in range(0, len(weights)):
                print(str(weights[l][k][j][i]), " ")

len(weights[0][0][0])
real_wgts = model_discrete.layers[3].get_weights()[0]

s = [1 for i in range(0, 25)]
s.extend([0 for i in range(0, 2375)])

model_discrete.layers[3].get_weights()[0].T[0]

fake_wgts = model_discrete.layers[3].set_weights([np.asarray([2 for i in range(0, 2400)]).reshape(5, 5, 6, 16)])#, [0 for i in range(0, 16)])

outp = keras_get_layer_output(model_discrete, model_discrete.layers[4], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp.shape)

#print(outp.T[0])
for k in range(0, 16):
    for i in range(0, 8):
        for j in range(0, 8):
            print(outp.T[k][j][i][0], end=" ")
        print()
    print()
    print()
    for i in range(0, 8):
        for j in range(0, 8):
            print(outp.T[k][i][j][0], end=" ")
        print()
    print()
    print()

for i in range(0, 6):
    print("------------------------------------------------------------------------------------")
    for j in range(0, 16):
        print(real_wgts.T[j][i])
        

model_discrete.layers[3].set_weights([real_wgts])

rld1 = real_wgt_dense1[0]

rld1.shape

arr = []
for i in range(0, 120):
    a = []
    for j in range(0, 256):
        a.append(rld1[j][i])
    arr.append(a)

arr = np.asarray(arr)

print(arr.shape)

for i in range(0, 120):
    arr[i] = np.reshape(arr[i], (1, 4, 4, 16)).T.flatten()

fh = open("tst.txt", "w")
for i in arr.flatten():
    fh.write(str(i)+" ")
fh.close()

model_discrete.layers[7].set_weights(real_wgt_dense1)

print(arr[0])

not_flat = keras_get_layer_output(model_discrete, model_discrete.layers[5], np.reshape(x_test[0], (1, 28, 28, 1)))
outp = keras_get_layer_output(model_discrete, model_discrete.layers[6], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp.shape)

print(outp[0])
print(not_flat)

fixed_order = []
for i in range(0, 16):
    fixed_order.extend(not_flat.T[i].T.flatten())
    
fixed_order = np.asarray(fixed_order)

for i in range(0, 120):
    a = arr[i].reshape(1, 4, 4, 16)
    b = []
    for j in range(0, 16):
        b.extend(a.T[j].T.flatten())
    arr[i] = np.asarray(b)

print(sum(arr[1]*fixed_order))

fh = open("test.txt", "w")
for i in arr.flatten():
    fh.write(str(i)+" ")
fh.close()

outp = keras_get_layer_output(model_discrete, model_discrete.layers[10], np.reshape(x_test[0], (1, 28, 28, 1)))
print(outp)



