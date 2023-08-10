from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import sklearn.metrics as sk

import pandas as pd
from collections import Counter
import numpy as np
import nltk

# import matplotlib.pyplot as plt
# import seaborn
# %matplotlib inline

modern = pd.read_pickle('data/5color_modern_no_name_hardmode.pkl')
Counter(modern.colors)

vectorizer = CountVectorizer()

y = pd.get_dummies(modern.colors)

X = vectorizer.fit_transform(modern.text)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=42)

xTrain = np.asarray(xTrain.todense())
xTest  = np.asarray(xTest.todense())
yTrain = np.asarray(yTrain)
yTest  = np.asarray(yTest)


print "There are {:,} words in the vocabulary.".format(len(vectorizer.vocabulary_))

get_ipython().run_cell_magic('time', '', '\n""" 5-layer """\n\n# batch normalization code adapted from \n# https://groups.google.com/forum/#!topic/theano-users/dMV6aabL1Ds \n\n\nimport theano\nfrom theano import tensor as T\nfrom theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams\nfrom theano.tensor.nnet.bn import batch_normalization\nimport numpy as np\n\nsrng = RandomStreams()\n\ndef shuffle(x, y):\n    # helper function to shuffle indicies each loop \n    index = np.random.choice(len(x), len(x), replace=False)\n    return x[index], y[index]\n\ndef floatX(X):\n    return np.asarray(X, dtype=theano.config.floatX)\n\ndef init_weights(shape):\n    (h, w) = shape\n    # Glorot normalization - last factor depends on non-linearity\n    # 0.25 for sigmoid and 0.1 for softmax, 1.0 for tanh or Relu\n    normalizer = 2.0 * np.sqrt(6) / np.sqrt(h + w) * 1.0\n    return theano.shared(floatX((np.random.random_sample(shape) - 0.5) * normalizer))\n\ndef rectify(X, alpha=0.01):\n#     return T.maximum(X, 0.)\n#    return T.maximum(X, 0.1*X)  #leaky rectifier\n     return T.switch(X > 0, X, alpha * (T.exp(X) - 1)) # ELU\n\ndef softmax(X):\n    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, \'x\'))\n    return e_x / e_x.sum(axis=1).dimshuffle(0, \'x\')\n\ndef RMSprop(cost, params, lr=0.001, rho=0.99, epsilon=1e-9):\n    grads = T.grad(cost=cost, wrt=params)\n    updates = []\n    for p, g in zip(params, grads):\n        acc = theano.shared(p.get_value() * 0.)\n        acc_new = rho * acc + (1 - rho) * g ** 2\n        gradient_scaling = T.sqrt(acc_new + epsilon)\n        g = g / gradient_scaling\n        updates.append((acc, acc_new))\n        updates.append((p, p - lr * g))\n    return updates\n\ndef model(X, w_h, g_h, bb_h, w_h2, g_h2, bb_h2,\n          w_h3, g_h3, bb_h3, w_o, g_ho, bb_ho):\n    \n    X = T.dot(X, w_h) \n    X = batch_normalization(X, gamma= g_h, beta= bb_h, \n                            mean= X.mean((0,), keepdims=True),\n                            std= T.ones_like(X.var((0,), keepdims = True)), \n                            mode=\'high_mem\') \n    h = rectify(X)\n\n    h  = T.dot(h, w_h2)\n    h = batch_normalization(h, gamma= g_h2, beta= bb_h2, \n                            mean= h.mean((0,), keepdims=True),\n                            std= T.ones_like(h.var((0,), keepdims = True)), \n                            mode=\'high_mem\') \n    h2 = rectify(h)\n\n    h2 = T.dot(h2, w_h3)\n    h2 = batch_normalization(h2, gamma= g_h3, beta= bb_h3, \n                            mean= h2.mean((0,), keepdims=True),\n                            std= T.ones_like(h2.var((0,), keepdims = True)), \n                            mode=\'high_mem\') \n    h3 = rectify(h2)\n    \n    h3 = T.dot(h3, w_o)\n    h3 = batch_normalization(h3, gamma= g_ho, beta= bb_ho, \n                            mean= h3.mean((0,), keepdims=True),\n                            std= T.ones_like(h3.var((0,), keepdims = True)), \n                            mode=\'high_mem\') \n    py_x = softmax(h3)\n    return h, h2, h3, py_x\n\n\nX = T.fmatrix()\nY = T.fmatrix()\n\nbatch_size = 60\n\nh1_size = 1000\nh2_size = 1000\nh3_size = 1000\n\nw_h = init_weights((len(vectorizer.vocabulary_), h1_size))\ng_h = theano.shared(floatX(np.ones((h1_size))))\nbb_h = theano.shared(floatX(np.zeros((h1_size))))\n\nw_h2 = init_weights((h1_size, h2_size))\ng_h2 = theano.shared(floatX(np.ones((h2_size))))\nbb_h2 = theano.shared(floatX(np.zeros((h2_size))))\n\nw_h3 = init_weights((h2_size, h3_size))\ng_h3 = theano.shared(floatX(np.ones((h3_size))))\nbb_h3 = theano.shared(floatX(np.zeros((h3_size))))\n\nw_o = init_weights((h3_size, yTest.shape[1]))\ng_ho = theano.shared(floatX(np.ones((yTest.shape[1]))))\nbb_ho = theano.shared(floatX(np.zeros((yTest.shape[1]))))\n\nnoise_h, noise_h2, noise_h3, noise_py_x = model(X, w_h, g_h, bb_h, \n                                      w_h2, g_h2, bb_h2, \n                                       w_h3, g_h3, bb_h3, \n                                      w_o, g_ho, bb_ho)\n\nh, h2, h3, py_x = model(X, w_h, g_h, bb_h, \n                    w_h2, g_h2, bb_h2, \n                     w_h3, g_h3, bb_h3, \n                    w_o, g_ho, bb_ho)\n\ny_x = T.argmax(py_x, axis=1)\n\n\ncost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))\nparams = [w_h, g_h, bb_h, w_h2, g_h2, bb_h2, \n           w_h3, g_h3, bb_h3, w_o, g_ho, bb_ho]\nupdates = RMSprop(cost, params, lr=0.0001)\n\ntrain = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)\npredict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)\n\n\nfor i in range(11):\n\n    for start, end in zip(range(0, len(xTrain), batch_size), range(batch_size, len(xTrain), batch_size)):\n        cost = train(xTrain[start:end], yTrain[start:end])\n        \n    xTrain, yTrain = shuffle(xTrain, yTrain)\n    xTest, yTest   = shuffle(xTest, yTest)\n\n    trr, tr = [], []\n    for start, end in zip(range(0, len(xTrain), batch_size), range(batch_size, len(xTrain), batch_size)):        \n        trr += [np.argmax(yTrain[start:end], axis=1) == predict(xTrain[start:end])]\n\n    for start, end in zip(range(0, len(xTest), batch_size), range(batch_size, len(xTest), batch_size)):\n        tr += [np.argmax(yTest[start:end], axis=1) == predict(xTest[start:end])]\n\n    print "Round: %-5s Test: %-14s Train: %-8s" % (i, np.mean(tr), np.mean(trr))\n    \nprint\n')







