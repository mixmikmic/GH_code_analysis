#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""i_rnn.py: test rnn code"""
from __future__ import print_function
from six.moves import xrange

import numpy as np
import sys
import copy    # for deepcopy

__author__    = "Shin Asakawa"
__copyright__ = "Copyright 2017, Tokyo JAPAN"
__credits__   = ["Shin Asakawa"]
__license__   = "MIT"
__version__   = "0.1"
__maintainer__ = "Shin Asakawa"
__email__      = "asakawa@ieee.org"

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def d_sigmoid(x):
    return x * (1.-x)

int2binary = {}
binary_dim = 8

# 2 の binary_dim 乗の数，2 のべき乗だから，2進数にすれば桁数のこと
largest_number = pow(2,binary_dim)
# 10進数を2進数に展開して array に格納
binary = np.unpackbits(np.array([range(largest_number)],
                                dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

## このセルは実行とは無関係。 np.unpackbits の確認用
np.unpackbits(np.array([range(4)],dtype=np.uint8).T, axis=1)
# np.array([range(3)])

#int2binary[3]
np.unpackbits(np.array([1],dtype=np.uint8))

### ハイパーパラメータの設定
### hyperparameters
lr = 0.1     # learning ratio 学習係数
MaxIter = 10000  # Maximum iteration number 繰り返しの最大数
N_i = 2      # number of units in the input layer  # 入力層のニューロン数(固定)
N_h = 16      # number of units in the hidden layer # 中間層のニューロン数(可変)
N_o = 1      # number of units in the output layer # 出力層のニューロン数(固定)

W_ih = 2 * np.random.random((N_i, N_h)) - 1.  # weight matrix from input to hidden
W_ho = 2 * np.random.random((N_h, N_o)) - 1.  # weight matrix from hidden to output
W_hh = 2 * np.random.random((N_h, N_h)) - 1.  # weight matrix from context to hidden

dW_ih = np.zeros_like(W_ih)    # delta matrix of W_ih
dW_ho = np.zeros_like(W_ho)    # delta matrix of W_ho
dW_hh = np.zeros_like(W_hh)    # delta matrix of W_hh

### 以下のループの意味
a = int2binary[3]
b = int2binary[4]
X = np.array([[a, b]])
print('X=',X)
c = int2binary[7]
# print(c)
Y = np.array([c]).T
print('Y=', Y)
print('for loop counter:', binary_dim)

# 具体例
X = np.array([[a[3], b[3]]])
X

for position in range(binary_dim):
    print(binary_dim - position - 1, end=' ')
print()    

for iter in range(MaxIter):
    # a と b を乱数で作り出して2進数に変換して格納
    a_int = np.random.randint(largest_number / 2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]

    # c にはその和を格納
    c_int = a_int + b_int
    c = int2binary[c_int]

    # d は 0 クリア
    d = np.zeros_like(c)

    total_err = 0
    Output_deltas = list()
    Hidden_values = list()
    Hidden_values.append(np.zeros(N_h))

    # forward prop
    for position in range(binary_dim):
        # a, b の上位桁から取り出して X に代入
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # y に c の上位桁から代入
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 入力行列 X に W_ih をかけて，Hidden_values かける W_hh をたして
        # ここに文を挿入
        
        Delta = y - Output
        # ここに文を挿入

        ### prediction
        d[binary_dim - position - 1] = np.round(Output[0][0])
        Hidden_values.append(copy.deepcopy(Hidden))
        #print('position=%d' % (position), end=' ')
        #print(Hidden_values)

    future_Hidden_delta = np.zeros(N_h)

    # back prop
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        Hidden       = Hidden_values[-position - 1]
        prev_Hidden  = Hidden_values[-position - 2]

        Output_delta = Output_deltas[-position - 1]
        Hidden_delta = (future_Hidden_delta.dot(W_hh.T) + Output_delta.dot(W_ho.T)) * d_sigmoid(Hidden)

        # ここに文を挿入
        future_Hiddne_delta = Hidden_delta

    W_ho += lr * dW_ho
    W_hh += lr * dW_hh
    W_ih += lr * dW_ih

    dW_ho *= 0
    dW_hh *= 0
    dW_ih *= 0

    if iter % 1000 == 0:
#    if iter % 5000 == 0:
        print("iteration=",iter)
        print("Error: %f" % (total_err))
        print("Pred: %s" % (str(d)))
        print("True: %s" % (str(c)))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))



