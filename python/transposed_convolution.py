import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.layers import Conv2D
from keras.models import Sequential

get_ipython().run_line_magic('matplotlib', 'inline')

inputs = np.random.randint(1, 9, size=(4, 4))
inputs

def show_matrix(m, color, cmap, title=None):
    rows, cols = len(m), len(m[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_yticks(list(range(rows)))
    ax.set_xticks(list(range(cols)))
    ax.xaxis.tick_top()  
    if title is not None:
        ax.set_title('{} {}'.format(title, m.shape), y=-0.5/rows)
    plt.imshow(m, cmap=cmap, vmin=0, vmax=1)
    for r in range(rows):
        for c in range(cols):
            text = '{:>3}'.format(int(m[r][c]))
            ax.text(c-0.2, r+0.15, text, color=color, fontsize=15)
    plt.show()
    
def show_inputs(m, title='Inputs'):
    show_matrix(m, 'b', plt.cm.Vega10, title)
    
def show_kernel(m, title='Kernel'):
    show_matrix(m, 'r', plt.cm.RdBu_r, title)
    
def show_output(m, title='Output'):
    show_matrix(m, 'g', plt.cm.GnBu, title)

show_inputs(inputs)

show_inputs(np.random.randint(100, 255, size=(4, 4)))

kernel = np.random.randint(1, 5, size=(3, 3))
kernel

show_kernel(kernel)

def convolve(m, k):
    m_rows, m_cols = len(m), len(m[0]) # matrix rows, cols
    k_rows, k_cols = len(k), len(k[0]) # kernel rows, cols

    rows = m_rows - k_rows + 1 # result matrix rows
    cols = m_rows - k_rows + 1 # result matrix cols
    
    v = np.zeros((rows, cols), dtype=m.dtype) # result matrix
    
    for r in range(rows):
        for c in range(cols):
            v[r][c] = np.sum(m[r:r+k_rows, c:c+k_cols] * k) # sum of the element-wise multiplication
    return v

output = convolve(inputs, kernel)
output

show_output(output)

output[0][0]

inputs[0:3, 0:3]

kernel

np.sum(inputs[0:3, 0:3] * kernel) # sum of the element-wise multiplication

def convolution_matrix(m, k):
    m_rows, m_cols = len(m), len(m[0]) # matrix rows, cols
    k_rows, k_cols = len(k), len(k[0]) # kernel rows, cols

    # output matrix rows and cols
    rows = m_rows - k_rows + 1 
    cols = m_rows - k_rows + 1
    
    # convolution matrix
    v = np.zeros((rows*cols, m_rows, m_cols)) 

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            v[i][r:r+k_rows, c:c+k_cols] = k

    v = v.reshape((rows*cols), -1)
    return v, rows, cols

C, rows, cols = convolution_matrix(inputs, kernel)

show_kernel(C, 'Convolution Matrix')

def column_vector(m):
    return m.flatten().reshape(-1, 1)

x = column_vector(inputs)
x

show_inputs(x)

output = C @ x
output

show_output(output)

output = output.reshape(rows, cols)
output

show_output(output)

show_kernel(C.T, 'Transposed Convolution Matrix')

x2 = np.random.randint(1, 5, size=(4, 1))
x2

show_inputs(x2)

output2 = (C.T @ x2)
output2

show_output(output2)

output2 = output2.reshape(4, 4)
output2

show_output(output2)

