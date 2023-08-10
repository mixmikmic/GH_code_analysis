try:
    if __IPYTHON__:
        from IPython import get_ipython

        get_ipython().magic('matplotlib inline')
        from ipython_utilities import *
        from ipywidgets import interact, fixed, FloatSlider, IntSlider, Label, Checkbox, FloatRangeSlider
        from IPython.display import display

        in_ipython_flag = True
except:
    in_ipython_flag = False
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from ipywidgets import interact, fixed, FloatSlider, IntSlider, Label, Checkbox, FloatRangeSlider

get_ipython().magic('matplotlib inline')
get_ipython().magic('pdb')

def display_two_sequences(n1,n2):
    index1 = np.linspace(0,15,n1)
    index2 = np.linspace(0,15,n2)
    A = 5*np.sin(index1)
    B = 3*np.sin(index2 + 1)
    # s1 = [1, 2, 3, 4]
    # s2 = [2, 3, 5, 6, 8]
    # ob_dtw = cl_dtw()
    # distance,_ = ob_dtw.calculate_dtw_distance(s1, s2)

    fig = plt.figure(figsize=(12,4))
    plt.plot(index1, A, '-ro', label='A')
    plt.plot(index2, B, '-bo' ,label='B')
    plt.ylabel('value')
    plt.xlabel('index')
    plt.legend()

    plt.show()
    plt.pause(0.001)
    
interact(display_two_sequences,
            n1=IntSlider(min=0, 
              max=100, step=1,value=5,
              description='# of points in sequence 1',
              continuous_update=True),
            n2=IntSlider(min=0, 
              max=100, step=1,value=7,
              description='# of points in sequence 2',
              continuous_update=True));

# arrange_widgets_in_grid(controls)

def dtw(s1, s2, window=3):
    grid = np.inf*np.ones((len(s1), len(s2)))
    # grid[0, :] = abs(s1[0] - s2)
    for i in range(window+1):
        grid[0, i] = abs(s1[0] - s2[i])
    for j in range(window+1):
        grid[j, 0] = abs(s2[0] - s1[j])
    
    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if abs(i-j) > window:
                continue
            grid[i, j] = abs(s1[i] - s2[j]) + min(grid[i - 1, j], grid[i, j-1], grid[i-1, j-1])
            
    
    print(grid)
    print(grid[-1, -1])
    
def display_two_sequences(n1,n2):
    index1 = np.linspace(0,15,n1)
    index2 = np.linspace(0,15,n2)
    A = [5, 6, 9, 2, 6]*2
    B = [5, 7, 2, 6, 9 , 2]*2
    #A = 5*np.sin(index1)
    #B = 3*np.sin(index2 + 1)
    # s1 = [1, 2, 3, 4]
    # s2 = [2, 3, 5, 6, 8]
    # ob_dtw = cl_dtw()
    # distance,_ = ob_dtw.calculate_dtw_distance(s1, s2)
    print(A)
    print(B)
    
    dtw(A, B)
#     fig = plt.figure(figsize=(12,4))
#     plt.plot(index1, A, '-ro', label='A')
#     plt.plot(index2, B, '-bo' ,label='B')
#     plt.ylabel('value')
#     plt.xlabel('index')
#     plt.legend()

#     plt.show()
#     plt.pause(0.001)
controls = interact(display_two_sequences,
             n1=IntSlider(min=0, 
            max=100, step=1,value=5,
            description='# of points in sequence 1',
            continuous_update=True),
                      n2=IntSlider(min=0, 
            max=100, step=1,value=7,
            description='# of points in sequence 2',
            continuous_update=True));

# arrange_widgets_in_grid(controls)



